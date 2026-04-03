"""
Tool-call specific repair logic for broken OpenAI-compatible responses.

Handles edge cases that providers like LLMZone get wrong when proxying
Anthropic/other models through an OpenAI-compatible interface.
"""

import json
import re
import uuid
import logging
from typing import Any

logger = logging.getLogger("proxy.tool_fixer")


def remove_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks (including multi-line) and orphaned tags.

    Args:
        text: Input string that may contain think tags.

    Returns:
        Cleaned string with all think-related tags removed.
    """
    if not isinstance(text, str):
        return text

    # Remove complete <think>...</think> blocks (greedy=False to handle multiple)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove orphaned opening tags
    cleaned = re.sub(r"<think>", "", cleaned, flags=re.IGNORECASE)

    # Remove orphaned closing tags
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)

    return cleaned


def sanitize_arguments(args_str: str) -> str:
    """
    Clean up a function arguments string into valid JSON.

    Steps:
    1. Strip whitespace
    2. Remove think tags
    3. Fix trailing commas
    4. Fix single quotes → double quotes
    5. Fix unquoted keys
    6. Fallback to {"raw": <original>} or "{}"

    Args:
        args_str: Raw arguments string from a tool call.

    Returns:
        Valid JSON string, or "{}" as last-resort fallback.
    """
    if not isinstance(args_str, str):
        return "{}"

    original = args_str
    text = args_str.strip()

    if not text:
        return "{}"

    # Remove think tags
    text = remove_think_tags(text).strip()

    if not text:
        return "{}"

    # Already valid? Return early.
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Fix 1: trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        json.loads(fixed)
        logger.debug("sanitize_arguments: fixed trailing comma in args")
        return fixed
    except json.JSONDecodeError:
        pass

    # Fix 2: single quotes → double quotes (naïve but handles common cases)
    try:
        # Replace single-quoted strings, being careful with escaped chars
        squote_fixed = re.sub(
            r"(?<![\\])'((?:[^'\\]|\\.)*)'",
            lambda m: '"' + m.group(1).replace('\\"', '"').replace('"', '\\"') + '"',
            fixed,
        )
        json.loads(squote_fixed)
        logger.debug("sanitize_arguments: fixed single quotes in args")
        return squote_fixed
    except (json.JSONDecodeError, Exception):
        pass

    # Fix 3: unquoted keys — wrap bare word keys in double quotes
    try:
        unquoted_fixed = re.sub(
            r"(?<![\"'\w])(\b[A-Za-z_][A-Za-z0-9_]*\b)\s*:",
            r'"\1":',
            fixed,
        )
        json.loads(unquoted_fixed)
        logger.debug("sanitize_arguments: fixed unquoted keys in args")
        return unquoted_fixed
    except (json.JSONDecodeError, Exception):
        pass

    # Fix 4: combine trailing-comma fix with unquoted-key fix
    try:
        combo = re.sub(r",\s*([}\]])", r"\1", text)
        combo = re.sub(
            r"(?<![\"'\w])(\b[A-Za-z_][A-Za-z0-9_]*\b)\s*:",
            r'"\1":',
            combo,
        )
        json.loads(combo)
        logger.debug("sanitize_arguments: fixed trailing comma + unquoted keys in args")
        return combo
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: wrap in {"raw": "..."}
    try:
        wrapped = json.dumps({"raw": original})
        logger.debug('sanitize_arguments: wrapping unparseable args in {"raw": ...}')
        return wrapped
    except Exception:
        return "{}"


def fix_tool_calls_response(response: dict) -> list[str]:
    """
    Apply tool-call specific fixes to a non-streaming response dict in-place.

    Fixes applied:
    A) Duplicate tool call IDs
    B) Empty/null arguments
    C) Malformed JSON in arguments
    D) Think tags in content/arguments
    E) Mixed content + tool_calls (finish_reason correction)
    F) Nested tool_calls unwrapping
    G) Wrong function.name namespace prefixes

    Args:
        response: OpenAI-compatible response dict (modified in-place).

    Returns:
        List of human-readable fix descriptions applied.
    """
    fixes: list[str] = []

    try:
        choices = response.get("choices")
        if not isinstance(choices, list):
            return fixes

        for choice_idx, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue

            message = choice.get("message")
            if not isinstance(message, dict):
                continue

            tool_calls = message.get("tool_calls")

            # --- Fix F: Nested tool_calls unwrapping ---
            if isinstance(tool_calls, dict) and "tool_calls" in tool_calls:
                logger.debug(
                    "fix_tool_calls_response [choice %d]: unwrapping nested tool_calls",
                    choice_idx,
                )
                message["tool_calls"] = tool_calls["tool_calls"]
                tool_calls = message["tool_calls"]
                fixes.append(f"choice[{choice_idx}]: unwrapped nested tool_calls")

            # Re-read after potential fix
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                continue

            # --- Fix D (content): Remove think tags from message.content ---
            content = message.get("content")
            if isinstance(content, str):
                cleaned_content = remove_think_tags(content)
                if cleaned_content != content:
                    logger.debug(
                        "fix_tool_calls_response [choice %d]: removed think tags from content",
                        choice_idx,
                    )
                    message["content"] = cleaned_content
                    fixes.append(
                        f"choice[{choice_idx}]: removed think tags from content"
                    )
                    content = cleaned_content

            # --- Fix E: Mixed content + tool_calls ---
            content = message.get("content")
            if isinstance(content, str):
                if content.strip() == "":
                    logger.debug(
                        "fix_tool_calls_response [choice %d]: setting whitespace-only content to None",
                        choice_idx,
                    )
                    message["content"] = None
                    fixes.append(
                        f"choice[{choice_idx}]: cleared whitespace-only content alongside tool_calls"
                    )
                # Always ensure finish_reason is "tool_calls" when tool_calls present
            if choice.get("finish_reason") != "tool_calls":
                logger.debug(
                    "fix_tool_calls_response [choice %d]: correcting finish_reason to 'tool_calls'",
                    choice_idx,
                )
                choice["finish_reason"] = "tool_calls"
                fixes.append(f"choice[{choice_idx}]: set finish_reason to 'tool_calls'")

            # --- Fix A: Duplicate tool call IDs ---
            seen_ids: set[str] = set()
            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if tc_id is None or tc_id in seen_ids:
                    new_id = f"call_{uuid.uuid4().hex[:24]}"
                    logger.debug(
                        "fix_tool_calls_response [choice %d, tc %d]: regenerating duplicate/missing id '%s' → '%s'",
                        choice_idx,
                        tc_idx,
                        tc_id,
                        new_id,
                    )
                    tc["id"] = new_id
                    fixes.append(
                        f"choice[{choice_idx}].tool_calls[{tc_idx}]: regenerated duplicate id '{tc_id}' → '{new_id}'"
                    )
                else:
                    seen_ids.add(tc_id)

                func = tc.get("function")
                if not isinstance(func, dict):
                    continue

                # --- Fix G: Wrong function.name namespace prefixes ---
                func_name = func.get("name", "")
                if isinstance(func_name, str):
                    for prefix in ("functions.", "tools."):
                        if func_name.startswith(prefix):
                            stripped = func_name[len(prefix) :]
                            logger.debug(
                                "fix_tool_calls_response [choice %d, tc %d]: stripping namespace prefix '%s' from '%s'",
                                choice_idx,
                                tc_idx,
                                prefix,
                                func_name,
                            )
                            func["name"] = stripped
                            fixes.append(
                                f"choice[{choice_idx}].tool_calls[{tc_idx}]: stripped namespace prefix '{prefix}' from function name"
                            )
                            break

                # --- Fix B: Empty/null arguments ---
                args = func.get("arguments")
                if args is None or (isinstance(args, str) and not args.strip()):
                    logger.debug(
                        "fix_tool_calls_response [choice %d, tc %d]: setting empty/null arguments to '{}'",
                        choice_idx,
                        tc_idx,
                    )
                    func["arguments"] = "{}"
                    fixes.append(
                        f"choice[{choice_idx}].tool_calls[{tc_idx}]: set empty/null arguments to '{{}}'"
                    )
                    continue

                if not isinstance(args, str):
                    # Non-string args (e.g., already a dict) — serialize
                    try:
                        func["arguments"] = json.dumps(args)
                        fixes.append(
                            f"choice[{choice_idx}].tool_calls[{tc_idx}]: serialized non-string arguments to JSON"
                        )
                    except Exception:
                        func["arguments"] = "{}"
                        fixes.append(
                            f"choice[{choice_idx}].tool_calls[{tc_idx}]: replaced unserializable arguments with '{{}}'"
                        )
                    continue

                # --- Fix D (arguments): Remove think tags ---
                cleaned_args = remove_think_tags(args)
                if cleaned_args != args:
                    logger.debug(
                        "fix_tool_calls_response [choice %d, tc %d]: removed think tags from arguments",
                        choice_idx,
                        tc_idx,
                    )
                    fixes.append(
                        f"choice[{choice_idx}].tool_calls[{tc_idx}]: removed think tags from arguments"
                    )
                    args = cleaned_args

                # --- Fix C: Malformed JSON in arguments ---
                try:
                    json.loads(args)
                    # Valid — only update if think tags were removed
                    if cleaned_args != func["arguments"]:
                        func["arguments"] = args
                except json.JSONDecodeError:
                    sanitized = sanitize_arguments(args)
                    logger.debug(
                        "fix_tool_calls_response [choice %d, tc %d]: sanitized malformed arguments",
                        choice_idx,
                        tc_idx,
                    )
                    func["arguments"] = sanitized
                    fixes.append(
                        f"choice[{choice_idx}].tool_calls[{tc_idx}]: sanitized malformed JSON arguments"
                    )

    except Exception as exc:
        logger.exception("fix_tool_calls_response: unexpected error: %s", exc)

    return fixes


def fix_streaming_tool_calls(
    chunks: list[dict],
) -> tuple[list[dict], list[str]]:
    """
    Analyze and fix tool-call issues in a collected list of streaming chunks.

    Fixes applied:
    - finish_reason "stop" → "tool_calls" when tool_calls appear in stream
    - Missing `index` fields on tool_call deltas
    - Ensure first delta for each tool_call carries its id

    Args:
        chunks: All collected streaming response chunks (modified in-place where needed).

    Returns:
        Tuple of (fixed_chunks, list_of_fix_descriptions).
    """
    fixes: list[str] = []

    if not isinstance(chunks, list) or not chunks:
        return chunks, fixes

    try:
        has_tool_calls = False
        final_finish_chunk_idx: int | None = None

        # --- Pass 1: detect tool_calls presence and locate finish chunk ---
        for chunk_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta", {})
                tc = delta.get("tool_calls") if isinstance(delta, dict) else None
                if isinstance(tc, list) and len(tc) > 0:
                    has_tool_calls = True

                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    final_finish_chunk_idx = chunk_idx

        if not has_tool_calls:
            return chunks, fixes

        # --- Fix: finish_reason "stop" → "tool_calls" ---
        if final_finish_chunk_idx is not None:
            finish_chunk = chunks[final_finish_chunk_idx]
            choices = finish_chunk.get("choices", [])
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") == "stop":
                    logger.debug(
                        "fix_streaming_tool_calls [chunk %d]: changing finish_reason 'stop' → 'tool_calls'",
                        final_finish_chunk_idx,
                    )
                    choice["finish_reason"] = "tool_calls"
                    fixes.append(
                        f"chunk[{final_finish_chunk_idx}]: changed finish_reason 'stop' → 'tool_calls'"
                    )

        # --- Pass 2: collect all unique indices, then re-index to 0-based ---
        # First collect every unique index we see (in encounter order)
        seen_indices_ordered: list[int] = []
        for chunk_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            for choice in chunk.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                for tc_delta in delta.get("tool_calls") or []:
                    if not isinstance(tc_delta, dict):
                        continue
                    idx = tc_delta.get("index")
                    if isinstance(idx, int) and idx not in seen_indices_ordered:
                        seen_indices_ordered.append(idx)

        # Build re-index map: old_index → new 0-based index
        reindex_map: dict[int, int] = {}
        needs_reindex = False
        for new_idx, old_idx in enumerate(sorted(seen_indices_ordered)):
            reindex_map[old_idx] = new_idx
            if old_idx != new_idx:
                needs_reindex = True

        if needs_reindex:
            fixes.append(f"streaming tool_call indices re-mapped: {reindex_map}")
            logger.debug(
                "fix_streaming_tool_calls: re-indexing tool_calls %s",
                reindex_map,
            )

        # --- Pass 3: fix indices, missing index, and track ids ---
        tc_counter = 0
        tc_id_by_index: dict[int, str | None] = {}
        first_delta_by_index: dict[
            int, int
        ] = {}  # new_index → chunk_idx of first occurrence

        for chunk_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices", [])
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                tool_calls = delta.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue

                for tc_delta in tool_calls:
                    if not isinstance(tc_delta, dict):
                        continue

                    # Fix missing index
                    if tc_delta.get("index") is None:
                        tc_delta["index"] = tc_counter
                        logger.debug(
                            "fix_streaming_tool_calls [chunk %d]: assigned missing index %d",
                            chunk_idx,
                            tc_counter,
                        )
                        fixes.append(
                            f"chunk[{chunk_idx}]: assigned index {tc_counter} to tool_call delta"
                        )
                        tc_counter += 1
                    elif needs_reindex:
                        old_idx = tc_delta["index"]
                        new_idx = reindex_map.get(old_idx, old_idx)
                        if old_idx != new_idx:
                            tc_delta["index"] = new_idx
                        if new_idx >= tc_counter:
                            tc_counter = new_idx + 1
                    else:
                        idx = tc_delta["index"]
                        if idx >= tc_counter:
                            tc_counter = idx + 1

                    idx: int = tc_delta["index"]

                    # Track id availability
                    tc_id = tc_delta.get("id")
                    if tc_id:
                        if idx not in tc_id_by_index:
                            tc_id_by_index[idx] = tc_id
                            first_delta_by_index[idx] = chunk_idx
                    elif idx not in first_delta_by_index:
                        first_delta_by_index[idx] = chunk_idx

        # Ensure first delta for each tool_call carries its id
        for chunk_idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices", [])
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                tool_calls = delta.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue

                for tc_delta in tool_calls:
                    if not isinstance(tc_delta, dict):
                        continue
                    raw_idx = tc_delta.get("index")
                    if not isinstance(raw_idx, int):
                        continue
                    tc_idx: int = raw_idx

                    is_first = first_delta_by_index.get(tc_idx) == chunk_idx
                    has_id = bool(tc_delta.get("id"))

                    if is_first and not has_id and tc_idx in tc_id_by_index:
                        tc_delta["id"] = tc_id_by_index[tc_idx]
                        logger.debug(
                            "fix_streaming_tool_calls [chunk %d]: moved id '%s' to first delta for index %d",
                            chunk_idx,
                            tc_id_by_index[tc_idx],
                            tc_idx,
                        )
                        fixes.append(
                            f"chunk[{chunk_idx}]: ensured id '{tc_id_by_index[tc_idx]}' present on first delta for tool_call index {tc_idx}"
                        )

    except Exception as exc:
        logger.exception("fix_streaming_tool_calls: unexpected error: %s", exc)

    return chunks, fixes
