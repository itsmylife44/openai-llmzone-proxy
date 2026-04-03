"""
OpenAI Responses API <-> Chat Completions API converter.

Translates:
  - Incoming Responses API requests (POST /v1/responses) into
    Chat Completions requests (POST /v1/chat/completions)
  - Chat Completions responses back into Responses API format

This allows clients that speak the newer Responses API (e.g. @ai-sdk/openai)
to work with upstreams that only support Chat Completions.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("proxy.responses_converter")


# ── Request: Responses API → Chat Completions ───────────────────────


def responses_to_chat_request(responses_req: dict) -> dict:
    """Convert a Responses API request into a Chat Completions request.

    Handles:
    - input (string or array) → messages
    - instructions → system message
    - tools (flat) → tools (wrapped in function)
    - max_output_tokens → max_tokens
    - text.format → response_format
    """
    chat_req: dict[str, Any] = {
        "model": responses_req.get("model", ""),
    }

    messages: list[dict] = []

    # ── System instruction ──────────────────────────────────────────
    instructions = responses_req.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # ── Input → messages ────────────────────────────────────────────
    raw_input = responses_req.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                _convert_input_item(item, messages)
    elif raw_input is not None:
        messages.append({"role": "user", "content": str(raw_input)})

    if not messages:
        messages.append({"role": "user", "content": ""})

    chat_req["messages"] = messages

    # ── Tools (unwrap → wrap) ───────────────────────────────────────
    tools = responses_req.get("tools")
    if isinstance(tools, list) and tools:
        chat_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )
            else:
                # web_search, file_search etc. — pass through and hope upstream
                # understands, otherwise it'll error
                chat_tools.append(tool)
        if chat_tools:
            chat_req["tools"] = chat_tools

    # ── tool_choice ─────────────────────────────────────────────────
    tc = responses_req.get("tool_choice")
    if tc is not None:
        chat_req["tool_choice"] = tc

    # ── Parameter renaming ──────────────────────────────────────────
    _RENAME = {
        "max_output_tokens": "max_tokens",
    }
    _PASSTHROUGH = [
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
    ]
    for rkey, ckey in _RENAME.items():
        if rkey in responses_req:
            chat_req[ckey] = responses_req[rkey]
    for key in _PASSTHROUGH:
        if key in responses_req:
            chat_req[key] = responses_req[key]

    # ── Streaming ───────────────────────────────────────────────────
    if responses_req.get("stream"):
        chat_req["stream"] = True

    # ── Structured output ───────────────────────────────────────────
    text_cfg = responses_req.get("text")
    if isinstance(text_cfg, dict):
        fmt = text_cfg.get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            chat_req["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", "response"),
                    "schema": fmt.get("schema", {}),
                },
            }

    logger.debug(
        "responses_to_chat_request: converted %d input items → %d messages",
        len(raw_input) if isinstance(raw_input, list) else 1,
        len(messages),
    )

    return chat_req


def _convert_input_item(item: dict, messages: list[dict]) -> None:
    """Convert a single Responses API input item into chat messages."""
    item_type = item.get("type") or item.get("role")

    if item_type == "message" or item.get("role") in ("user", "assistant", "system"):
        role = item.get("role", "user")
        content = item.get("content", "")

        # Content can be string or array of content parts
        if isinstance(content, list):
            # Convert content parts
            chat_content = []
            for part in content:
                if isinstance(part, str):
                    chat_content.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    ptype = part.get("type", "text")
                    if ptype == "input_text":
                        chat_content.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "input_image":
                        chat_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.get("image_url", part.get("url", ""))},
                            }
                        )
                    elif ptype == "text":
                        chat_content.append({"type": "text", "text": part.get("text", "")})
                    else:
                        chat_content.append(part)
            messages.append({"role": role, "content": chat_content})
        else:
            messages.append({"role": role, "content": str(content)})

    elif item_type == "tool_result" or item_type == "function_call_output":
        # Tool result → tool message
        messages.append(
            {
                "role": "tool",
                "tool_call_id": item.get("call_id", item.get("tool_call_id", "")),
                "content": item.get("output", item.get("content", "")),
            }
        )

    elif item_type == "function_call":
        # This is a previous assistant tool call — reconstruct as assistant message
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:24]}")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                ],
            }
        )

    else:
        # Unknown type — try as user message
        content = item.get("content") or item.get("text") or str(item)
        messages.append({"role": "user", "content": str(content)})


# ── Response: Chat Completions → Responses API ──────────────────────


def chat_to_responses_response(chat_resp: dict, original_input: Any = None) -> dict:
    """Convert a Chat Completions response into a Responses API response.

    Maps:
    - choices[0].message.content → output[{type: "message", content: [...]}]
    - choices[0].message.tool_calls → output[{type: "function_call", ...}]
    - finish_reason → status
    - usage field names
    """
    resp_id = chat_resp.get("id", f"resp_{uuid.uuid4().hex[:32]}")
    # Strip chatcmpl- prefix for responses format
    if resp_id.startswith("chatcmpl-"):
        resp_id = f"resp_{resp_id[9:]}"

    model = chat_resp.get("model", "unknown")
    created = chat_resp.get("created", int(time.time()))

    choice = {}
    if isinstance(chat_resp.get("choices"), list) and chat_resp["choices"]:
        choice = chat_resp["choices"][0]

    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    # ── Build output items ──────────────────────────────────────────
    output: list[dict] = []

    # Text content → message output item
    content = message.get("content")
    if content is not None and content != "":
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )

    # Tool calls → function_call output items
    tool_calls = message.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            output.append(
                {
                    "type": "function_call",
                    "id": (tc_id := tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
                    "call_id": tc_id,
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                    "status": "completed",
                }
            )

    # ── Map finish_reason → status ──────────────────────────────────
    status_map = {
        "stop": "completed",
        "tool_calls": "completed",
        "length": "incomplete",
        "content_filter": "failed",
    }
    status = status_map.get(finish_reason, "completed")

    # ── Usage ───────────────────────────────────────────────────────
    chat_usage = chat_resp.get("usage", {})
    usage = {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "total_tokens": chat_usage.get("total_tokens", 0),
    }

    responses_resp = {
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "model": model,
        "output": output,
        "status": status,
        "usage": usage,
        "metadata": {},
        "temperature": None,
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "truncation": "disabled",
        "incomplete_details": None,
        "error": None,
    }

    return responses_resp


# ── Streaming: Chat Completions chunks → Responses API SSE events ────


class ResponsesStreamState:
    """Tracks state across a streaming Responses API conversion session."""

    def __init__(self, response_id: str, model: str = "unknown") -> None:
        self.response_id = response_id
        self.model = model
        self.msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.text_emitted = False
        self.created_emitted = False
        self.output_item_added: set[int] = set()  # track which output_index got .added
        self.collected_text = ""
        self.collected_tool_calls: dict[int, dict] = {}  # idx → {id, name, args}


def chat_chunk_to_responses_events(
    chunk: dict,
    *,
    state: ResponsesStreamState,
    output_index: int = 0,
) -> list[str]:
    """Convert a Chat Completions streaming chunk into Responses API SSE events.

    Returns a list of SSE-formatted strings.  Each is ``data: {json}\\n\\n``
    with the ``type`` field **inside** the JSON (no ``event:`` header).
    This is the format expected by ``@ai-sdk/openai``.
    """
    events: list[str] = []

    def _emit(obj: dict) -> None:
        events.append(f"data: {json.dumps(obj)}\n\n")

    model = chunk.get("model", state.model)
    state.model = model

    choices = chunk.get("choices", [])
    if not choices:
        return events

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # ── First chunk: response.created + response.in_progress ────────
    if not state.created_emitted:
        state.created_emitted = True
        resp_obj = _make_response_obj(state, status="in_progress")
        _emit({"type": "response.created", "response": resp_obj})
        _emit({"type": "response.in_progress", "response": resp_obj})

    # ── Content delta ───────────────────────────────────────────────
    content = delta.get("content")
    if content:
        # First text → emit output_item.added + content_part.added
        if output_index not in state.output_item_added:
            state.output_item_added.add(output_index)
            msg_item = {
                "type": "message",
                "id": state.msg_item_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            _emit(
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": msg_item,
                }
            )
            _emit(
                {
                    "type": "response.content_part.added",
                    "output_index": output_index,
                    "content_index": 0,
                    "item_id": state.msg_item_id,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            )

        state.collected_text += content
        _emit(
            {
                "type": "response.output_text.delta",
                "output_index": output_index,
                "content_index": 0,
                "item_id": state.msg_item_id,
                "delta": content,
            }
        )

    # ── Tool call deltas ────────────────────────────────────────────
    tc_deltas = delta.get("tool_calls")
    if isinstance(tc_deltas, list):
        for tc_d in tc_deltas:
            if not isinstance(tc_d, dict):
                continue
            tc_idx = tc_d.get("index", 0)
            tc_id = tc_d.get("id")
            func = tc_d.get("function", {})
            tc_name = func.get("name")
            tc_args = func.get("arguments", "")

            # Track this tool call
            if tc_idx not in state.collected_tool_calls:
                state.collected_tool_calls[tc_idx] = {
                    "id": tc_id or f"call_{uuid.uuid4().hex[:24]}",
                    "name": tc_name or "",
                    "arguments": "",
                }
            tc_state = state.collected_tool_calls[tc_idx]
            if tc_id:
                tc_state["id"] = tc_id
            if tc_name:
                tc_state["name"] = tc_name

            oi = output_index + tc_idx

            # First appearance → output_item.added
            if oi not in state.output_item_added:
                state.output_item_added.add(oi)
                fc_item = {
                    "type": "function_call",
                    "id": tc_state["id"],
                    "call_id": tc_state["id"],
                    "name": tc_state["name"],
                    "arguments": "",
                    "status": "in_progress",
                }
                _emit(
                    {
                        "type": "response.output_item.added",
                        "output_index": oi,
                        "item": fc_item,
                    }
                )

            # Arguments delta
            if tc_args:
                tc_state["arguments"] += tc_args
                _emit(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": oi,
                        "item_id": tc_state["id"],
                        "delta": tc_args,
                    }
                )

    # ── Finish ──────────────────────────────────────────────────────
    if finish_reason:
        # Emit output_item.done for each item
        # Text message
        if state.collected_text:
            _emit(
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": state.msg_item_id,
                    "text": state.collected_text,
                }
            )
            _emit(
                {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": state.msg_item_id,
                    "part": {
                        "type": "output_text",
                        "text": state.collected_text,
                        "annotations": [],
                    },
                }
            )
            done_item = {
                "type": "message",
                "id": state.msg_item_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": state.collected_text,
                        "annotations": [],
                    }
                ],
            }
            _emit(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": done_item,
                }
            )

        # Function calls
        for tc_idx in sorted(state.collected_tool_calls.keys()):
            tc_s = state.collected_tool_calls[tc_idx]
            oi = output_index + tc_idx
            _emit(
                {
                    "type": "response.function_call_arguments.done",
                    "output_index": oi,
                    "item_id": tc_s["id"],
                    "arguments": tc_s["arguments"],
                }
            )
            done_fc = {
                "type": "function_call",
                "id": tc_s["id"],
                "call_id": tc_s["id"],
                "name": tc_s["name"],
                "arguments": tc_s["arguments"],
                "status": "completed",
            }
            _emit(
                {
                    "type": "response.output_item.done",
                    "output_index": oi,
                    "item": done_fc,
                }
            )

        # response.completed
        status_map = {
            "stop": "completed",
            "tool_calls": "completed",
            "length": "incomplete",
            "content_filter": "failed",
        }
        status = status_map.get(finish_reason, "completed")
        resp_obj = _make_response_obj(state, status=status)
        event_type = "response.completed" if status == "completed" else f"response.{status}"
        _emit({"type": event_type, "response": resp_obj})

    return events


def _make_response_obj(state: ResponsesStreamState, status: str = "in_progress") -> dict:
    """Build a response object for created/completed events."""
    output: list[dict] = []
    if state.collected_text:
        output.append(
            {
                "type": "message",
                "id": state.msg_item_id,
                "status": "completed" if status == "completed" else "in_progress",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": state.collected_text,
                        "annotations": [],
                    }
                ],
            }
        )
    for tc_idx in sorted(state.collected_tool_calls.keys()):
        tc_s = state.collected_tool_calls[tc_idx]
        output.append(
            {
                "type": "function_call",
                "id": tc_s["id"],
                "call_id": tc_s["id"],
                "name": tc_s["name"],
                "arguments": tc_s["arguments"],
                "status": "completed" if status == "completed" else "in_progress",
            }
        )

    return {
        "id": state.response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": state.model,
        "output": output,
        "status": status,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
