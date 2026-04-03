"""
Streaming handler functions for the proxy.

Handles both Chat Completions and Responses API streaming (SSE) requests.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from copy import deepcopy
from typing import AsyncGenerator

import httpx
from fastapi.responses import StreamingResponse

from proxy import config
from proxy.core.key_manager import KeyManager
from proxy.core.request_log import log_streaming_summary
from proxy.core.response_validator import validate_chunk
from proxy.core.responses_converter import chat_chunk_to_responses_events, ResponsesStreamState
from proxy.core.tool_call_fixer import fix_streaming_tool_calls
from proxy.handlers.non_streaming import report_upstream_error, report_upstream_success

logger = logging.getLogger("proxy")


# ── Helpers ─────────────────────────────────────────────────────────


def remap_tool_call_index(
    tc_d: dict,
    tc_index_map: dict[int, int],
    tc_next_index: list[int],
) -> None:
    """Re-map a tool_call delta's index from upstream (possibly 1-based) to 0-based."""
    if isinstance(tc_d, dict) and isinstance(tc_d.get("index"), int):
        old_idx = tc_d["index"]
        if old_idx not in tc_index_map:
            tc_index_map[old_idx] = tc_next_index[0]
            tc_next_index[0] += 1
        new_idx = tc_index_map[old_idx]
        if old_idx != new_idx:
            tc_d["index"] = new_idx


# ── Streaming handlers ──────────────────────────────────────────────


async def handle_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    body: dict,
    t0: float,
    http_client: httpx.AsyncClient,
    key_manager: KeyManager | None,
    api_key: str | None = None,
) -> StreamingResponse:
    """Proxy SSE stream with on-the-fly validation.

    Strategy:
    - Stream chunks through to the client in real-time for low latency.
    - Validate each chunk individually (structure, object type).
    - Collect all chunks in memory to detect tool-call issues at the end.
    - If tool-call fixes are needed, we inject a corrected final chunk
      before sending [DONE].
    """

    async def _stream_generator() -> AsyncGenerator[bytes, None]:
        collected_chunks: list[dict] = []
        had_tool_calls = False
        final_finish_reason: str | None = None
        chunk_count = 0
        # Track tool_call index re-mapping: upstream_index → 0-based index
        tc_index_map: dict[int, int] = {}
        tc_next_index: list[int] = [0]

        try:
            async with http_client.stream(
                "POST",
                upstream_url,
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    # Read error body and yield as SSE error
                    error_body = await resp.aread()
                    report_upstream_error(
                        key_manager,
                        api_key,
                        resp.status_code,
                        error_body.decode("utf-8", errors="replace")[:500],
                    )
                    logger.error(f"[{request_id}] Upstream error {resp.status_code}: {error_body[:500]}")
                    error_data = {
                        "error": {
                            "message": f"Upstream error: {resp.status_code}",
                            "type": "upstream_error",
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

                buffer = ""
                async for raw_bytes in resp.aiter_bytes():
                    buffer += raw_bytes.decode("utf-8", errors="replace")

                    # Process complete SSE lines from buffer
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line == "data: [DONE]":
                            # Before sending DONE, check if we need streaming fixes
                            if collected_chunks and config.ENABLE_TOOL_FIXER:
                                fixed_chunks, stream_fixes = fix_streaming_tool_calls(collected_chunks)

                                if stream_fixes:
                                    logger.info(f"[{request_id}] Streaming fixes applied: {stream_fixes}")
                                    # Send corrected final chunk if finish_reason was fixed
                                    for fc in fixed_chunks:
                                        choices = fc.get("choices", [])
                                        if choices:
                                            fr = choices[0].get("finish_reason")
                                            if fr == "tool_calls" and final_finish_reason != "tool_calls":
                                                yield f"data: {json.dumps(fc)}\n\n".encode()
                                                final_finish_reason = "tool_calls"

                                    if config.ENABLE_REQUEST_LOGGING:
                                        duration_ms = (time.monotonic() - t0) * 1000
                                        log_streaming_summary(
                                            request_id,
                                            chunk_count,
                                            stream_fixes,
                                            duration_ms,
                                            had_tool_calls,
                                            final_finish_reason,
                                        )

                            yield b"data: [DONE]\n\n"

                            # Stream completed successfully
                            report_upstream_success(key_manager, api_key)

                            if config.ENABLE_REQUEST_LOGGING and not had_tool_calls:
                                duration_ms = (time.monotonic() - t0) * 1000
                                log_streaming_summary(
                                    request_id,
                                    chunk_count,
                                    [],
                                    duration_ms,
                                    had_tool_calls,
                                    final_finish_reason,
                                )
                            return

                        if not line.startswith("data: "):
                            # Forward non-data SSE lines (comments, event types) as-is
                            yield f"{line}\n\n".encode()
                            continue

                        json_str = line[6:]  # Strip "data: " prefix

                        # ── Parse chunk ─────────────────────────────
                        try:
                            chunk_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            # Forward unparseable chunks as-is
                            logger.warning(f"[{request_id}] Unparseable chunk: {json_str[:200]}")
                            yield f"data: {json_str}\n\n".encode()
                            continue

                        chunk_count += 1

                        # ── Validate chunk structure ────────────────
                        if config.ENABLE_RESPONSE_VALIDATOR:
                            chunk_fixes = validate_chunk(chunk_data)
                            if chunk_fixes:
                                logger.debug(f"[{request_id}] Chunk {chunk_count} fixes: {chunk_fixes}")

                        # ── Re-map tool_call indices to 0-based ────
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            fr = choices[0].get("finish_reason")

                            tc_deltas = delta.get("tool_calls")
                            if isinstance(tc_deltas, list):
                                had_tool_calls = True
                                for tc_d in tc_deltas:
                                    remap_tool_call_index(tc_d, tc_index_map, tc_next_index)

                            if fr is not None:
                                # ── Fix: LLMZone sends finish_reason="tool_calls"
                                # even when NO tool_calls were in the stream.
                                # Correct to "stop" if we never saw any tool_calls.
                                if fr == "tool_calls" and not had_tool_calls:
                                    logger.debug(
                                        f"[{request_id}] LLMZone bug: finish_reason='tool_calls' "
                                        f"but no tool_calls in stream. Correcting to 'stop'."
                                    )
                                    choices[0]["finish_reason"] = "stop"
                                    fr = "stop"
                                final_finish_reason = fr

                        # ── Collect for post-stream analysis ────────
                        collected_chunks.append(deepcopy(chunk_data))

                        # ── Forward chunk ───────────────────────────
                        yield f"data: {json.dumps(chunk_data)}\n\n".encode()

                # Handle remaining buffer (edge case: no trailing newline)
                if buffer.strip():
                    if buffer.strip() == "data: [DONE]":
                        yield b"data: [DONE]\n\n"
                    elif buffer.strip().startswith("data: "):
                        yield f"{buffer.strip()}\n\n".encode()

        except httpx.TimeoutException:
            duration_ms = (time.monotonic() - t0) * 1000
            logger.error(f"[{request_id}] Stream timeout after {duration_ms:.0f}ms")
            error_chunk = {"error": {"message": "Upstream stream timeout", "type": "timeout_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        except Exception as exc:
            logger.exception(f"[{request_id}] Stream error: {exc}")
            error_chunk = {"error": {"message": "Proxy internal error", "type": "proxy_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def handle_responses_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    chat_body: dict,
    original_body: dict,
    t0: float,
    http_client: httpx.AsyncClient,
    key_manager: KeyManager | None,
    api_key: str | None = None,
) -> StreamingResponse:
    """Stream Chat Completions chunks, translate each into Responses API SSE events."""
    resp_id = f"resp_{uuid.uuid4().hex[:32]}"

    async def _stream_gen() -> AsyncGenerator[bytes, None]:
        stream_state = ResponsesStreamState(response_id=resp_id)
        tc_index_map: dict[int, int] = {}
        tc_next_index: list[int] = [0]

        try:
            async with http_client.stream(
                "POST",
                upstream_url,
                json=chat_body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    report_upstream_error(
                        key_manager,
                        api_key,
                        resp.status_code,
                        error_body.decode("utf-8", errors="replace")[:500],
                    )
                    logger.error(f"[{request_id}] Upstream error {resp.status_code}: {error_body[:500]}")
                    error_event = {
                        "type": "response.failed",
                        "response": {
                            "id": resp_id,
                            "object": "response",
                            "status": "failed",
                            "error": {"message": f"Upstream error: {resp.status_code}"},
                        },
                    }
                    yield f"data: {json.dumps(error_event)}\n\n".encode()
                    yield b"data: [DONE]\n\n"
                    return

                buffer = ""
                chunk_count = 0

                async for raw_bytes in resp.aiter_bytes():
                    buffer += raw_bytes.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue

                        if line == "data: [DONE]":
                            yield b"data: [DONE]\n\n"
                            report_upstream_success(key_manager, api_key)
                            duration_ms = (time.monotonic() - t0) * 1000
                            logger.info(
                                f"[{request_id}] Responses API stream complete: "
                                f"{chunk_count} chunks, {duration_ms:.0f}ms"
                            )
                            return

                        if not line.startswith("data: "):
                            continue

                        try:
                            chunk_data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        chunk_count += 1

                        # Validate the Chat Completions chunk
                        if config.ENABLE_RESPONSE_VALIDATOR:
                            validate_chunk(chunk_data)

                        # Re-map tool_call indices to 0-based
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            tc_deltas = delta.get("tool_calls")
                            if isinstance(tc_deltas, list):
                                for tc_d in tc_deltas:
                                    remap_tool_call_index(tc_d, tc_index_map, tc_next_index)

                        # Convert to Responses API events
                        events = chat_chunk_to_responses_events(
                            chunk_data,
                            state=stream_state,
                            output_index=0,
                        )

                        for event_str in events:
                            yield event_str.encode()

                # Handle remaining buffer
                if buffer.strip() == "data: [DONE]":
                    yield b"data: [DONE]\n\n"

        except httpx.TimeoutException:
            error_event = {
                "type": "response.failed",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "status": "failed",
                    "error": {"message": "Upstream timeout"},
                },
            }
            yield f"data: {json.dumps(error_event)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as exc:
            logger.exception(f"[{request_id}] Responses stream error: {exc}")
            error_event = {
                "type": "response.failed",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "status": "failed",
                    "error": {"message": "Proxy internal error"},
                },
            }
            yield f"data: {json.dumps(error_event)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        _stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
