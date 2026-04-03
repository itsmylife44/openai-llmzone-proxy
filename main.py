"""
OpenAI-compatible proxy server for LLMZone.

Sits between OpenCode (or any OpenAI-compatible client) and the upstream
LLM provider.  Validates and fixes responses to ensure strict OpenAI API
conformance — especially around tool_calls / function calling.

Supports both streaming (SSE) and non-streaming modes.
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

import config
from middleware.logging import (
    generate_request_id,
    log_request,
    log_response,
    log_streaming_summary,
    setup_logging,
)
from middleware.response_validator import validate_response, validate_chunk
from middleware.tool_call_fixer import (
    fix_tool_calls_response,
    fix_streaming_tool_calls,
)
from middleware.responses_converter import (
    responses_to_chat_request,
    chat_to_responses_response,
    chat_chunk_to_responses_events,
    ResponsesStreamState,
)
from middleware.key_manager import KeyManager

# ── Logging ─────────────────────────────────────────────────────────
setup_logging(config.LOG_LEVEL)
logger = logging.getLogger("proxy")

# ── Key Manager (rotation + blacklisting) ──────────────────────────
_key_manager: KeyManager | None = None
if config.UPSTREAM_API_KEYS:
    _key_manager = KeyManager(
        keys=config.UPSTREAM_API_KEYS,
        cooldown_seconds=config.KEY_COOLDOWN_SECONDS,
    )
    logger.info(
        "Key rotation enabled: %d key(s), cooldown=%ds",
        _key_manager.key_count,
        config.KEY_COOLDOWN_SECONDS,
    )
else:
    logger.info("Key rotation disabled: no API keys configured")

# ── HTTP client (shared, connection-pooled) ─────────────────────────
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.UPSTREAM_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        follow_redirects=True,
    )
    logger.info("HTTP client initialized")
    yield
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        logger.info("HTTP client closed")


# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLMZone OpenAI Proxy",
    description="OpenAI-compatible proxy with response validation & tool-call fixing",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _upstream_url(path: str) -> str:
    """Build the full upstream URL for a given path."""
    base = config.UPSTREAM_BASE_URL.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


def _get_api_key() -> str | None:
    """Get the next available API key from the key manager.

    Returns the key string, or None if all keys are blacklisted.
    """
    if _key_manager is not None:
        return _key_manager.get_key()
    # Fallback: static key from config
    return config.UPSTREAM_API_KEY or None


def _upstream_headers(request: Request, api_key: str | None = None) -> dict[str, str]:
    """Build headers for the upstream request.

    Uses the provided API key (from key manager), or falls back to
    the client-supplied Authorization header (pass-through mode).
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Pass through the client's auth header
        auth = request.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth

    return headers


def _report_upstream_error(
    api_key: str | None,
    status_code: int,
    error_body: str = "",
) -> None:
    """Report an upstream error to the key manager for potential blacklisting."""
    if _key_manager is not None and api_key is not None:
        _key_manager.report_error(
            key=api_key,
            status_code=status_code,
            error_body=error_body,
        )


def _report_upstream_success(api_key: str | None) -> None:
    """Report a successful upstream response to the key manager."""
    if _key_manager is not None and api_key is not None:
        _key_manager.report_success(api_key)


def _apply_fixes(response_data: dict) -> list[str]:
    """Run the full validation + fix pipeline on a response dict."""
    all_fixes: list[str] = []

    if config.ENABLE_RESPONSE_VALIDATOR:
        all_fixes.extend(validate_response(response_data))

    if config.ENABLE_TOOL_FIXER:
        all_fixes.extend(fix_tool_calls_response(response_data))

    return all_fixes


# ── Health check ────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    result: dict = {
        "status": "ok",
        "upstream": config.UPSTREAM_BASE_URL,
        "features": {
            "response_validator": config.ENABLE_RESPONSE_VALIDATOR,
            "tool_fixer": config.ENABLE_TOOL_FIXER,
            "request_logging": config.ENABLE_REQUEST_LOGGING,
            "key_rotation": config.ENABLE_KEY_ROTATION,
        },
    }
    if _key_manager is not None:
        result["keys"] = {
            "total": _key_manager.key_count,
            "active": _key_manager.active_key_count,
        }
    return result


# ── Chat completions (main proxy endpoint) ─────────────────────────


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    request_id = generate_request_id()
    t0 = time.monotonic()

    # ── Parse request body ──────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
        )

    # ── Log request ─────────────────────────────────────────────────
    if config.ENABLE_REQUEST_LOGGING:
        log_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            headers=dict(request.headers),
            body=body,
        )

    is_streaming = body.get("stream", False)

    # ── Get API key (rotation-aware) ────────────────────────────────
    api_key = _get_api_key()
    if api_key is None and _key_manager is not None:
        # All keys blacklisted
        logger.error(f"[{request_id}] All API keys are blacklisted!")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "All upstream API keys are currently rate-limited. Please try again later.",
                    "type": "rate_limit_error",
                }
            },
        )

    # ── Build upstream request ──────────────────────────────────────
    upstream_url = _upstream_url("/chat/completions")
    headers = _upstream_headers(request, api_key)

    try:
        if is_streaming:
            return await _handle_streaming(request_id, upstream_url, headers, body, t0, api_key)
        else:
            return await _handle_non_streaming(request_id, upstream_url, headers, body, t0, api_key)
    except httpx.TimeoutException:
        duration = (time.monotonic() - t0) * 1000
        logger.error(f"[{request_id}] Upstream timeout after {duration:.0f}ms")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Upstream provider timeout",
                    "type": "timeout_error",
                }
            },
        )
    except httpx.ConnectError as exc:
        logger.error(f"[{request_id}] Upstream connection error: {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Cannot connect to upstream: {exc}",
                    "type": "connection_error",
                }
            },
        )
    except Exception as exc:
        logger.exception(f"[{request_id}] Unexpected error")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Proxy error: {exc}", "type": "proxy_error"}},
        )


# ── Non-streaming handler ──────────────────────────────────────────


async def _handle_non_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    body: dict,
    t0: float,
    api_key: str | None = None,
) -> JSONResponse:
    assert _http_client is not None
    resp = await _http_client.post(upstream_url, json=body, headers=headers)
    duration_ms = (time.monotonic() - t0) * 1000

    # ── Parse upstream response ─────────────────────────────────────
    try:
        upstream_data = resp.json()
    except Exception:
        logger.error(f"[{request_id}] Upstream returned non-JSON: {resp.text[:500]}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Upstream returned invalid JSON",
                    "type": "proxy_error",
                }
            },
        )

    # ── If upstream returned an error, report to key manager & pass through
    if resp.status_code >= 400:
        error_text = json.dumps(upstream_data) if upstream_data else ""
        _report_upstream_error(api_key, resp.status_code, error_text)
        if config.ENABLE_REQUEST_LOGGING:
            log_response(
                request_id,
                resp.status_code,
                upstream_data,
                upstream_data,
                [],
                duration_ms,
            )
        return JSONResponse(status_code=resp.status_code, content=upstream_data)

    # ── Success: report to key manager ──────────────────────────────
    _report_upstream_success(api_key)

    # ── Validate & fix ──────────────────────────────────────────────
    upstream_raw = deepcopy(upstream_data)
    fixes = _apply_fixes(upstream_data)

    # ── Log response ────────────────────────────────────────────────
    if config.ENABLE_REQUEST_LOGGING:
        log_response(
            request_id,
            resp.status_code,
            upstream_raw,
            upstream_data,
            fixes,
            duration_ms,
        )

    if fixes:
        logger.info(f"[{request_id}] Applied {len(fixes)} fixes: {fixes}")

    return JSONResponse(status_code=200, content=upstream_data)


# ── Streaming handler ───────────────────────────────────────────────


async def _handle_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    body: dict,
    t0: float,
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
        assert _http_client is not None
        collected_chunks: list[dict] = []
        had_tool_calls = False
        final_finish_reason: str | None = None
        chunk_count = 0
        # Track tool_call index re-mapping: upstream_index → 0-based index
        tc_index_map: dict[int, int] = {}
        tc_next_index: int = 0

        try:
            async with _http_client.stream(
                "POST",
                upstream_url,
                json=body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    # Read error body and yield as SSE error
                    error_body = await resp.aread()
                    _report_upstream_error(
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
                            _report_upstream_success(api_key)

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
                                    if isinstance(tc_d, dict) and isinstance(tc_d.get("index"), int):
                                        old_idx = tc_d["index"]
                                        if old_idx not in tc_index_map:
                                            tc_index_map[old_idx] = tc_next_index
                                            tc_next_index += 1
                                        new_idx = tc_index_map[old_idx]
                                        if old_idx != new_idx:
                                            tc_d["index"] = new_idx

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


# ── Responses API endpoint (translate to Chat Completions) ──────────


@app.post("/v1/responses")
@app.post("/responses")
async def responses_api(request: Request) -> Response:
    """Handle OpenAI Responses API requests by translating them to Chat Completions."""
    request_id = generate_request_id()
    t0 = time.monotonic()

    # ── Parse request body ──────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
        )

    logger.info(f"[{request_id}] Responses API request → translating to Chat Completions")

    if config.ENABLE_REQUEST_LOGGING:
        log_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            headers=dict(request.headers),
            body=body,
        )

    # ── Convert Responses API → Chat Completions ────────────────────
    try:
        chat_body = responses_to_chat_request(body)
    except Exception as exc:
        logger.error(f"[{request_id}] Failed to convert Responses API request: {exc}")
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Request conversion error: {exc}",
                    "type": "invalid_request_error",
                }
            },
        )

    logger.debug(
        f"[{request_id}] Converted request: model={chat_body.get('model')} "
        f"messages={len(chat_body.get('messages', []))} "
        f"tools={len(chat_body.get('tools', []))} "
        f"stream={chat_body.get('stream', False)}"
    )

    is_streaming = chat_body.get("stream", False)
    upstream_url = _upstream_url("/chat/completions")

    # ── Get API key (rotation-aware) ────────────────────────────────
    api_key = _get_api_key()
    if api_key is None and _key_manager is not None:
        logger.error(f"[{request_id}] All API keys are blacklisted!")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": "All upstream API keys are currently rate-limited. Please try again later.",
                    "type": "rate_limit_error",
                }
            },
        )

    headers = _upstream_headers(request, api_key)

    try:
        if is_streaming:
            return await _handle_responses_streaming(
                request_id,
                upstream_url,
                headers,
                chat_body,
                body,
                t0,
                api_key,
            )
        else:
            return await _handle_responses_non_streaming(
                request_id,
                upstream_url,
                headers,
                chat_body,
                body,
                t0,
                api_key,
            )
    except httpx.TimeoutException:
        duration = (time.monotonic() - t0) * 1000
        logger.error(f"[{request_id}] Upstream timeout after {duration:.0f}ms")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Upstream provider timeout",
                    "type": "timeout_error",
                }
            },
        )
    except httpx.ConnectError as exc:
        logger.error(f"[{request_id}] Upstream connection error: {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Cannot connect to upstream: {exc}",
                    "type": "connection_error",
                }
            },
        )
    except Exception as exc:
        logger.exception(f"[{request_id}] Unexpected error in Responses API handler")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"Proxy error: {exc}", "type": "proxy_error"}},
        )


async def _handle_responses_non_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    chat_body: dict,
    original_body: dict,
    t0: float,
    api_key: str | None = None,
) -> JSONResponse:
    """Send as Chat Completions, convert response back to Responses API format."""
    assert _http_client is not None
    resp = await _http_client.post(upstream_url, json=chat_body, headers=headers)
    duration_ms = (time.monotonic() - t0) * 1000

    try:
        upstream_data = resp.json()
    except Exception:
        logger.error(f"[{request_id}] Upstream returned non-JSON: {resp.text[:500]}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Upstream returned invalid JSON",
                    "type": "proxy_error",
                }
            },
        )

    if resp.status_code >= 400:
        error_text = json.dumps(upstream_data) if upstream_data else ""
        _report_upstream_error(api_key, resp.status_code, error_text)
        if config.ENABLE_REQUEST_LOGGING:
            log_response(
                request_id,
                resp.status_code,
                upstream_data,
                upstream_data,
                [],
                duration_ms,
            )
        return JSONResponse(status_code=resp.status_code, content=upstream_data)

    # Report success
    _report_upstream_success(api_key)

    # ── Validate & fix the Chat Completions response ────────────────
    upstream_raw = deepcopy(upstream_data)
    fixes = _apply_fixes(upstream_data)

    if fixes:
        logger.info(f"[{request_id}] Applied {len(fixes)} Chat Completions fixes before conversion")

    # ── Convert Chat Completions → Responses API ────────────────────
    responses_data = chat_to_responses_response(upstream_data, original_body.get("input"))

    if config.ENABLE_REQUEST_LOGGING:
        log_response(
            request_id,
            resp.status_code,
            upstream_raw,
            responses_data,
            fixes,
            duration_ms,
        )

    return JSONResponse(status_code=200, content=responses_data)


async def _handle_responses_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    chat_body: dict,
    original_body: dict,
    t0: float,
    api_key: str | None = None,
) -> StreamingResponse:
    """Stream Chat Completions chunks, translate each into Responses API SSE events."""
    resp_id = f"resp_{uuid.uuid4().hex[:32]}"

    async def _stream_gen() -> AsyncGenerator[bytes, None]:
        assert _http_client is not None
        stream_state = ResponsesStreamState(response_id=resp_id)
        tc_index_map: dict[int, int] = {}
        tc_next_index: int = 0

        try:
            async with _http_client.stream(
                "POST",
                upstream_url,
                json=chat_body,
                headers=headers,
            ) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    _report_upstream_error(
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
                            _report_upstream_success(api_key)
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
                                    if isinstance(tc_d, dict) and isinstance(tc_d.get("index"), int):
                                        old_idx = tc_d["index"]
                                        if old_idx not in tc_index_map:
                                            tc_index_map[old_idx] = tc_next_index
                                            tc_next_index += 1
                                        tc_d["index"] = tc_index_map[old_idx]

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


# ── Models endpoint (pass-through) ─────────────────────────────────


@app.get("/v1/models")
@app.get("/models")
async def list_models(request: Request) -> Response:
    assert _http_client is not None
    headers = _upstream_headers(request)

    try:
        resp = await _http_client.get(_upstream_url("/models"), headers=headers)
        data = resp.json()
        return JSONResponse(status_code=resp.status_code, content=data)
    except Exception as exc:
        logger.error(f"Models endpoint error: {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Failed to fetch models: {exc}",
                    "type": "proxy_error",
                }
            },
        )


# ── Catch-all for other /v1/* endpoints ─────────────────────────────


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str) -> Response:
    """Forward any other /v1/* request to upstream as-is."""
    assert _http_client is not None
    headers = _upstream_headers(request)

    method = request.method.upper()
    upstream = _upstream_url(f"/{path}")

    try:
        if method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            resp = await _http_client.request(method, upstream, content=body, headers=headers)
        else:
            resp = await _http_client.request(method, upstream, headers=headers)

        # Try to return JSON, fall back to raw text
        try:
            data = resp.json()
            return JSONResponse(status_code=resp.status_code, content=data)
        except Exception:
            return Response(
                status_code=resp.status_code,
                content=resp.content,
                media_type=resp.headers.get("content-type", "application/octet-stream"),
            )
    except Exception as exc:
        logger.error(f"Catch-all error for {method} /v1/{path}: {exc}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Proxy error: {exc}", "type": "proxy_error"}},
        )


# ── Entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting proxy on {config.PROXY_HOST}:{config.PROXY_PORT} -> {config.UPSTREAM_BASE_URL}")
    uvicorn.run(
        "main:app",
        host=config.PROXY_HOST,
        port=config.PROXY_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
