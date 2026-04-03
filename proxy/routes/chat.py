"""
Chat Completions proxy endpoint.

Handles POST /v1/chat/completions and POST /chat/completions.
"""

from __future__ import annotations

import logging
import time

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from proxy import config
from proxy.core.request_log import generate_request_id, log_request
from proxy.handlers.non_streaming import handle_non_streaming
from proxy.handlers.streaming import handle_streaming
from proxy.routes._helpers import (
    get_api_key,
    upstream_headers,
    upstream_url,
)

logger = logging.getLogger("proxy")

router = APIRouter()


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    request_id = generate_request_id()
    t0 = time.monotonic()

    http_client: httpx.AsyncClient = request.app.state.http_client
    key_manager = request.app.state.key_manager

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
    api_key = get_api_key(key_manager)
    if api_key is None and key_manager is not None:
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
    url = upstream_url("/chat/completions")
    headers = upstream_headers(request, api_key)

    try:
        if is_streaming:
            return await handle_streaming(request_id, url, headers, body, t0, http_client, key_manager, api_key)
        else:
            return await handle_non_streaming(request_id, url, headers, body, t0, http_client, key_manager, api_key)
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
