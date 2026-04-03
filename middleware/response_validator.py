"""
OpenAI Chat Completions API response validator.

Ensures every response conforms to the OpenAI spec, fixing deviations in-place.
Works for both non-streaming (full response) and individual streaming chunks.
"""

import json
import time
import uuid
import re
import logging
from typing import Any

logger = logging.getLogger("proxy.validator")

# Valid finish reasons per OpenAI spec
_VALID_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter"}

# Top-level keys allowed in a non-streaming response
_ALLOWED_TOP_LEVEL_KEYS = {
    "id",
    "object",
    "created",
    "model",
    "choices",
    "usage",
    "system_fingerprint",
    "service_tier",
}

# Top-level keys allowed in a streaming chunk
_ALLOWED_CHUNK_TOP_LEVEL_KEYS = {
    "id",
    "object",
    "created",
    "model",
    "choices",
    "system_fingerprint",
    "service_tier",
    "usage",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_response(response: dict) -> list[str]:
    """Validate and fix a NON-STREAMING Chat Completions response in-place.

    Returns a list of human-readable descriptions of every fix applied.
    Never raises; all parsing is wrapped in try/except.
    """
    fixes: list[str] = []

    try:
        _stage1_structure(response, fixes)
        _stage2_choices_message(response, fixes)
        _stage3_tool_calls(response, fixes)
        _stage4_usage(response, fixes)
        _stage5_cleanup(response, fixes)
    except Exception as exc:  # pragma: no cover
        msg = f"Unexpected error during response validation: {exc}"
        logger.warning(msg)
        fixes.append(msg)

    return fixes


def validate_chunk(chunk: dict) -> list[str]:
    """Validate and fix a single STREAMING chunk in-place.

    Returns a list of human-readable descriptions of every fix applied.
    Never raises; all parsing is wrapped in try/except.
    """
    fixes: list[str] = []

    try:
        _chunk_stage1_structure(chunk, fixes)
        _chunk_stage2_choices_delta(chunk, fixes)
        _chunk_stage3_cleanup(chunk, fixes)
    except Exception as exc:  # pragma: no cover
        msg = f"Unexpected error during chunk validation: {exc}"
        logger.warning(msg)
        fixes.append(msg)

    return fixes


def extract_tool_calls_from_content(content: str) -> list[dict] | None:
    """Try to detect and extract tool calls embedded in plain-text content.

    Handles:
    - ``<tool_call>...</tool_call>`` XML-style tags (GLM / ChatGLM format)
    - ``<function_call>...</function_call>`` XML-style tags
    - JSON objects containing both ``"name"`` and ``"arguments"`` keys
    - ``{"function_call": {...}}`` wrapper patterns

    Returns a list of properly formatted tool_call dicts (OpenAI schema), or
    ``None`` if no tool calls could be detected.
    """
    if not content or not isinstance(content, str):
        return None

    tool_calls: list[dict] = []

    # --- Pattern 1: <tool_call>...</tool_call> tags ---
    xml_pattern = re.compile(
        r"<(?:tool_call|function_call)>(.*?)</(?:tool_call|function_call)>",
        re.DOTALL | re.IGNORECASE,
    )
    for match in xml_pattern.finditer(content):
        inner = match.group(1).strip()
        tc = _parse_tool_call_json(inner)
        if tc:
            tool_calls.append(tc)

    if tool_calls:
        return tool_calls

    # --- Pattern 2: {"function_call": {...}} wrapper ---
    fc_pattern = re.compile(
        r'\{\s*"function_call"\s*:\s*(\{.*?\})\s*\}',
        re.DOTALL,
    )
    for match in fc_pattern.finditer(content):
        try:
            inner_obj = json.loads(match.group(1))
            name = inner_obj.get("name") or inner_obj.get("function_name")
            arguments = inner_obj.get("arguments") or inner_obj.get("parameters") or {}
            if name:
                tool_calls.append(_make_tool_call(name, arguments))
        except (json.JSONDecodeError, AttributeError):
            pass

    if tool_calls:
        return tool_calls

    # --- Pattern 3: bare JSON objects with "name" + "arguments" keys ---
    # Find all top-level JSON objects in the string
    for obj in _iter_json_objects(content):
        try:
            if isinstance(obj, dict):
                name = obj.get("name")
                arguments = obj.get("arguments") or obj.get("parameters")
                if name and isinstance(name, str) and arguments is not None:
                    tool_calls.append(_make_tool_call(name, arguments))
        except Exception:
            pass

    return tool_calls if tool_calls else None


# ---------------------------------------------------------------------------
# Stage helpers — non-streaming
# ---------------------------------------------------------------------------


def _stage1_structure(response: dict, fixes: list[str]) -> None:
    """Stage 1: validate/fix top-level structural fields."""

    # id
    current_id = response.get("id")
    if not isinstance(current_id, str) or not current_id.startswith("chatcmpl-"):
        new_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        fix = f"id: replaced {current_id!r} → {new_id!r}"
        response["id"] = new_id
        _log_fix(fix, fixes)

    # object
    if response.get("object") != "chat.completion":
        fix = f"object: replaced {response.get('object')!r} → 'chat.completion'"
        response["object"] = "chat.completion"
        _log_fix(fix, fixes)

    # created
    current_created = response.get("created")
    if not isinstance(current_created, int):
        new_created = int(time.time())
        fix = f"created: replaced {current_created!r} → {new_created}"
        response["created"] = new_created
        _log_fix(fix, fixes)

    # model
    if not isinstance(response.get("model"), str):
        fix = f"model: replaced {response.get('model')!r} → 'unknown'"
        response["model"] = "unknown"
        _log_fix(fix, fixes)

    # choices
    if not isinstance(response.get("choices"), list):
        fix = "choices: was missing/non-list; injected default single-choice list"
        response["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ]
        _log_fix(fix, fixes)


def _stage2_choices_message(response: dict, fixes: list[str]) -> None:
    """Stage 2: validate/fix each choice and its message."""
    choices: list[Any] = response.get("choices", [])

    for enum_idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            choices[enum_idx] = {
                "index": enum_idx,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
            _log_fix(f"choices[{enum_idx}]: was not a dict; replaced with default", fixes)
            continue

        # index
        if not isinstance(choice.get("index"), int):
            fix = f"choices[{enum_idx}].index: replaced {choice.get('index')!r} → {enum_idx}"
            choice["index"] = enum_idx
            _log_fix(fix, fixes)

        # message
        if not isinstance(choice.get("message"), dict):
            fix = f"choices[{enum_idx}].message: was missing/non-dict; created empty dict"
            choice["message"] = {}
            _log_fix(fix, fixes)

        msg: dict = choice["message"]

        # message.role
        if msg.get("role") != "assistant":
            fix = f"choices[{enum_idx}].message.role: replaced {msg.get('role')!r} → 'assistant'"
            msg["role"] = "assistant"
            _log_fix(fix, fixes)

        # message.content
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            new_content = str(content)
            fix = f"choices[{enum_idx}].message.content: coerced {type(content).__name__} → str"
            msg["content"] = new_content
            _log_fix(fix, fixes)

        # finish_reason — initial pass (tool_calls stage may override)
        finish_reason = choice.get("finish_reason")
        if finish_reason not in _VALID_FINISH_REASONS:
            # Will be properly inferred in stage 3; set a safe default now
            choice["finish_reason"] = "stop"
            if finish_reason is not None:
                fix = (
                    f"choices[{enum_idx}].finish_reason: replaced invalid "
                    f"{finish_reason!r} → 'stop' (may be updated by stage 3)"
                )
                _log_fix(fix, fixes)


def _stage3_tool_calls(response: dict, fixes: list[str]) -> None:
    """Stage 3 (CRITICAL): validate/fix tool_calls in every choice."""
    choices: list[Any] = response.get("choices", [])

    for enum_idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue

        msg: dict = choice.get("message", {})
        tool_calls = msg.get("tool_calls")

        # --- Case A: tool_calls already present ---
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            _fix_tool_calls_list(tool_calls, enum_idx, fixes)

            # finish_reason MUST be "tool_calls"
            if choice.get("finish_reason") != "tool_calls":
                fix = (
                    f"choices[{enum_idx}].finish_reason: corrected "
                    f"{choice.get('finish_reason')!r} → 'tool_calls' "
                    f"(tool_calls present)"
                )
                choice["finish_reason"] = "tool_calls"
                _log_fix(fix, fixes)

            # content MUST be null when tool_calls are present
            if msg.get("content") == "":
                msg["content"] = None
                _log_fix(
                    f"choices[{enum_idx}].message.content: set '' → null (tool_calls present)",
                    fixes,
                )

        # --- Case B: no tool_calls — check if content hides them ---
        elif tool_calls is None or tool_calls == []:
            content = msg.get("content")
            if isinstance(content, str) and _looks_like_tool_call_content(content):
                extracted = None
                try:
                    extracted = extract_tool_calls_from_content(content)
                except Exception as exc:
                    logger.debug("Failed to extract tool calls from content: %s", exc)

                if extracted:
                    msg["tool_calls"] = extracted
                    msg["content"] = None
                    choice["finish_reason"] = "tool_calls"
                    fix = (
                        f"choices[{enum_idx}]: extracted {len(extracted)} tool call(s) "
                        f"from content; set content → null, finish_reason → 'tool_calls'"
                    )
                    _log_fix(fix, fixes)


def _stage4_usage(response: dict, fixes: list[str]) -> None:
    """Stage 4: ensure usage block is present and all fields are integers."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        response["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        _log_fix("usage: was missing/non-dict; injected zero-value usage block", fixes)
        return

    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        val = usage.get(field)
        if not isinstance(val, int):
            try:
                new_val = int(val) if val is not None else 0
            except (TypeError, ValueError):
                new_val = 0
            fix = f"usage.{field}: coerced {val!r} → {new_val}"
            usage[field] = new_val
            _log_fix(fix, fixes)


def _stage5_cleanup(response: dict, fixes: list[str]) -> None:
    """Stage 5: remove unexpected keys at all levels (logged, not silently dropped)."""

    # Top-level
    unexpected = [k for k in list(response.keys()) if k not in _ALLOWED_TOP_LEVEL_KEYS]
    for key in unexpected:
        fix = f"top-level key {key!r} is not in the OpenAI spec; removed"
        logger.debug(fix)
        fixes.append(fix)
        del response[key]

    # Nested: choices[].* and choices[].message.*
    _ALLOWED_CHOICE_KEYS = {"index", "message", "finish_reason", "logprobs"}
    _ALLOWED_MESSAGE_KEYS = {
        "role",
        "content",
        "tool_calls",
        "function_call",
        "refusal",
    }

    for choice_idx, choice in enumerate(response.get("choices", [])):
        if not isinstance(choice, dict):
            continue

        # Clean choice-level keys
        bad_choice_keys = [k for k in list(choice.keys()) if k not in _ALLOWED_CHOICE_KEYS]
        for key in bad_choice_keys:
            fix = f"choices[{choice_idx}].{key}: non-standard key removed"
            logger.debug(fix)
            fixes.append(fix)
            del choice[key]

        # Clean message-level keys
        msg = choice.get("message")
        if isinstance(msg, dict):
            bad_msg_keys = [k for k in list(msg.keys()) if k not in _ALLOWED_MESSAGE_KEYS]
            for key in bad_msg_keys:
                fix = f"choices[{choice_idx}].message.{key}: non-standard key removed"
                logger.debug(fix)
                fixes.append(fix)
                del msg[key]

            # Also clean up tool_calls: remove null value (should be absent, not null)
            if "tool_calls" in msg and msg["tool_calls"] is None:
                del msg["tool_calls"]

    # Nested: usage.* — allow standard + common extensions
    _ALLOWED_USAGE_KEYS = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    }
    usage = response.get("usage")
    if isinstance(usage, dict):
        bad_usage_keys = [k for k in list(usage.keys()) if k not in _ALLOWED_USAGE_KEYS]
        for key in bad_usage_keys:
            fix = f"usage.{key}: non-standard key removed"
            logger.debug(fix)
            fixes.append(fix)
            del usage[key]


# ---------------------------------------------------------------------------
# Stage helpers — streaming chunks
# ---------------------------------------------------------------------------


def _chunk_stage1_structure(chunk: dict, fixes: list[str]) -> None:
    """Validate/fix structural fields of a streaming chunk."""

    # id
    current_id = chunk.get("id")
    if not isinstance(current_id, str) or not current_id.startswith("chatcmpl-"):
        new_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        fix = f"chunk.id: replaced {current_id!r} → {new_id!r}"
        chunk["id"] = new_id
        _log_fix(fix, fixes)

    # object
    if chunk.get("object") != "chat.completion.chunk":
        fix = f"chunk.object: replaced {chunk.get('object')!r} → 'chat.completion.chunk'"
        chunk["object"] = "chat.completion.chunk"
        _log_fix(fix, fixes)

    # created
    current_created = chunk.get("created")
    if not isinstance(current_created, int):
        new_created = int(time.time())
        fix = f"chunk.created: replaced {current_created!r} → {new_created}"
        chunk["created"] = new_created
        _log_fix(fix, fixes)

    # model
    if not isinstance(chunk.get("model"), str):
        fix = f"chunk.model: replaced {chunk.get('model')!r} → 'unknown'"
        chunk["model"] = "unknown"
        _log_fix(fix, fixes)

    # choices
    if not isinstance(chunk.get("choices"), list):
        fix = "chunk.choices: was missing/non-list; injected empty list"
        chunk["choices"] = []
        _log_fix(fix, fixes)


def _chunk_stage2_choices_delta(chunk: dict, fixes: list[str]) -> None:
    """Validate/fix choices/delta in a streaming chunk."""
    choices: list[Any] = chunk.get("choices", [])

    for enum_idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            choices[enum_idx] = {"index": enum_idx, "delta": {}, "finish_reason": None}
            _log_fix(
                f"chunk.choices[{enum_idx}]: was not a dict; replaced with default",
                fixes,
            )
            continue

        # index
        if not isinstance(choice.get("index"), int):
            fix = f"chunk.choices[{enum_idx}].index: replaced {choice.get('index')!r} → {enum_idx}"
            choice["index"] = enum_idx
            _log_fix(fix, fixes)

        # delta
        if not isinstance(choice.get("delta"), dict):
            fix = f"chunk.choices[{enum_idx}].delta: was missing/non-dict; created empty dict"
            choice["delta"] = {}
            _log_fix(fix, fixes)

        delta: dict = choice["delta"]

        # delta.role — if present, must be "assistant"
        if "role" in delta and delta["role"] != "assistant":
            fix = f"chunk.choices[{enum_idx}].delta.role: replaced {delta['role']!r} → 'assistant'"
            delta["role"] = "assistant"
            _log_fix(fix, fixes)

        # delta.content — must be string or None if present
        if "content" in delta:
            content = delta["content"]
            if content is not None and not isinstance(content, str):
                delta["content"] = str(content)
                _log_fix(
                    f"chunk.choices[{enum_idx}].delta.content: coerced {type(content).__name__} → str",
                    fixes,
                )

        # delta.tool_calls — each entry must have an index field
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    continue
                if not isinstance(tc.get("index"), int):
                    fix = (
                        f"chunk.choices[{enum_idx}].delta.tool_calls[{tc_idx}].index: "
                        f"replaced {tc.get('index')!r} → {tc_idx}"
                    )
                    tc["index"] = tc_idx
                    _log_fix(fix, fixes)

                # type must be "function" if present
                if "type" in tc and tc["type"] != "function":
                    fix = (
                        f"chunk.choices[{enum_idx}].delta.tool_calls[{tc_idx}].type: "
                        f"replaced {tc['type']!r} → 'function'"
                    )
                    tc["type"] = "function"
                    _log_fix(fix, fixes)

                # function.arguments must be a string if present
                func = tc.get("function")
                if isinstance(func, dict) and "arguments" in func:
                    args = func["arguments"]
                    if args is not None and not isinstance(args, str):
                        try:
                            func["arguments"] = json.dumps(args)
                        except (TypeError, ValueError):
                            func["arguments"] = str(args)
                        fix = (
                            f"chunk.choices[{enum_idx}].delta.tool_calls[{tc_idx}]"
                            f".function.arguments: coerced {type(args).__name__} → str"
                        )
                        _log_fix(fix, fixes)

        # finish_reason — None until last chunk; if present must be valid or None
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and finish_reason not in _VALID_FINISH_REASONS:
            fix = f"chunk.choices[{enum_idx}].finish_reason: replaced invalid {finish_reason!r} → null"
            choice["finish_reason"] = None
            _log_fix(fix, fixes)


def _chunk_stage3_cleanup(chunk: dict, fixes: list[str]) -> None:
    """Remove unexpected keys from a streaming chunk at all levels."""

    # Top-level
    unexpected = [k for k in list(chunk.keys()) if k not in _ALLOWED_CHUNK_TOP_LEVEL_KEYS]
    for key in unexpected:
        fix = f"chunk top-level key {key!r} is not in the OpenAI spec; removed"
        logger.debug(fix)
        fixes.append(fix)
        del chunk[key]

    # Nested: choices[].* and choices[].delta.*
    _ALLOWED_CHUNK_CHOICE_KEYS = {"index", "delta", "finish_reason", "logprobs"}
    _ALLOWED_DELTA_KEYS = {"role", "content", "tool_calls", "function_call", "refusal"}

    for choice_idx, choice in enumerate(chunk.get("choices", [])):
        if not isinstance(choice, dict):
            continue

        bad_choice_keys = [k for k in list(choice.keys()) if k not in _ALLOWED_CHUNK_CHOICE_KEYS]
        for key in bad_choice_keys:
            fix = f"chunk.choices[{choice_idx}].{key}: non-standard key removed"
            logger.debug(fix)
            fixes.append(fix)
            del choice[key]

        delta = choice.get("delta")
        if isinstance(delta, dict):
            bad_delta_keys = [k for k in list(delta.keys()) if k not in _ALLOWED_DELTA_KEYS]
            for key in bad_delta_keys:
                fix = f"chunk.choices[{choice_idx}].delta.{key}: non-standard key removed"
                logger.debug(fix)
                fixes.append(fix)
                del delta[key]

            # Clean up null tool_calls in delta (should be absent, not null)
            if "tool_calls" in delta and delta["tool_calls"] is None:
                del delta["tool_calls"]

            # Clean up null content in delta (keep only meaningful values)
            if "content" in delta and delta["content"] is None:
                del delta["content"]

            # Clean up null role (only first chunk should have role)
            if "role" in delta and delta["role"] is None:
                del delta["role"]


# ---------------------------------------------------------------------------
# Tool-call helpers
# ---------------------------------------------------------------------------


def _fix_tool_calls_list(tool_calls: list, choice_idx: int, fixes: list[str]) -> None:
    """Fix individual tool_call entries inside a choices[n].message.tool_calls list."""
    for tc_idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            # Replace with a stub rather than crash
            tool_calls[tc_idx] = {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": "unknown", "arguments": "{}"},
            }
            _log_fix(
                f"choices[{choice_idx}].message.tool_calls[{tc_idx}]: was not a dict; replaced with stub",
                fixes,
            )
            continue

        # id
        if not isinstance(tc.get("id"), str) or not tc["id"]:
            new_id = f"call_{uuid.uuid4().hex[:24]}"
            fix = f"choices[{choice_idx}].message.tool_calls[{tc_idx}].id: replaced {tc.get('id')!r} → {new_id!r}"
            tc["id"] = new_id
            _log_fix(fix, fixes)

        # type
        if tc.get("type") != "function":
            fix = f"choices[{choice_idx}].message.tool_calls[{tc_idx}].type: replaced {tc.get('type')!r} → 'function'"
            tc["type"] = "function"
            _log_fix(fix, fixes)

        # function
        if not isinstance(tc.get("function"), dict):
            fix = f"choices[{choice_idx}].message.tool_calls[{tc_idx}].function: was missing/non-dict; created stub"
            tc["function"] = {"name": "unknown", "arguments": "{}"}
            _log_fix(fix, fixes)

        func: dict = tc["function"]

        # function.name
        if not isinstance(func.get("name"), str) or not func["name"]:
            fix = (
                f"choices[{choice_idx}].message.tool_calls[{tc_idx}].function.name: "
                f"replaced {func.get('name')!r} → 'unknown'"
            )
            func["name"] = "unknown"
            _log_fix(fix, fixes)

        # function.arguments — MUST be a JSON-encoded string
        args = func.get("arguments")
        if isinstance(args, (dict, list)):
            try:
                func["arguments"] = json.dumps(args)
            except (TypeError, ValueError):
                func["arguments"] = "{}"
            fix = (
                f"choices[{choice_idx}].message.tool_calls[{tc_idx}].function.arguments: "
                f"serialised {type(args).__name__} → JSON string"
            )
            _log_fix(fix, fixes)
        elif not isinstance(args, str):
            func["arguments"] = "{}"
            fix = f"choices[{choice_idx}].message.tool_calls[{tc_idx}].function.arguments: replaced {args!r} → '{{}}'"
            _log_fix(fix, fixes)
        else:
            # It's a string — make sure it's valid JSON; try to fix if not
            try:
                json.loads(args)
            except json.JSONDecodeError:
                fixed_args = _try_fix_json_string(args)
                if fixed_args is not None:
                    fix = (
                        f"choices[{choice_idx}].message.tool_calls[{tc_idx}]"
                        f".function.arguments: repaired invalid JSON string"
                    )
                    func["arguments"] = fixed_args
                    _log_fix(fix, fixes)
                else:
                    func["arguments"] = "{}"
                    fix = (
                        f"choices[{choice_idx}].message.tool_calls[{tc_idx}]"
                        f".function.arguments: could not repair JSON; replaced with '{{}}'"
                    )
                    _log_fix(fix, fixes)


def _make_tool_call(name: str, arguments: Any) -> dict:
    """Create a properly-shaped tool_call dict."""
    if isinstance(arguments, str):
        # Validate it's parseable; fall back to wrapping in a key
        try:
            json.loads(arguments)
            args_str = arguments
        except json.JSONDecodeError:
            args_str = json.dumps({"value": arguments})
    else:
        try:
            args_str = json.dumps(arguments)
        except (TypeError, ValueError):
            args_str = "{}"

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def _parse_tool_call_json(text: str) -> dict | None:
    """Try to parse a JSON blob into a tool_call dict."""
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None
        name = obj.get("name") or obj.get("function_name") or obj.get("tool_name")
        arguments = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if not name:
            return None
        return _make_tool_call(str(name), arguments)
    except (json.JSONDecodeError, AttributeError):
        return None


def _looks_like_tool_call_content(content: str) -> bool:
    """Quick heuristic check: does this content string look like it contains tool calls?"""
    patterns = [
        r"<tool_call\s*>",
        r"<function_call\s*>",
        r'"function_call"\s*:',
        r'"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:',
        r'"tool_name"\s*:\s*"[^"]+"',
    ]
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return True
    return False


def _iter_json_objects(text: str):
    """Yield all top-level JSON objects found in *text* (best-effort)."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = text[start : i + 1]
                try:
                    yield json.loads(fragment)
                except json.JSONDecodeError:
                    pass
                start = None


def _try_fix_json_string(s: str) -> str | None:
    """Attempt simple repairs on a broken JSON string. Returns fixed string or None."""
    # Try stripping outer quotes if someone double-encoded
    stripped = s.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            inner = json.loads(stripped)  # decode the string
            if isinstance(inner, str):
                json.loads(inner)  # check inner is valid JSON
                return inner
        except (json.JSONDecodeError, ValueError):
            pass

    # Try replacing single quotes with double quotes (common mistake)
    try:
        candidate = stripped.replace("'", '"')
        json.loads(candidate)
        return candidate
    except (json.JSONDecodeError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _log_fix(message: str, fixes: list[str]) -> None:
    logger.debug("validator fix: %s", message)
    fixes.append(message)
