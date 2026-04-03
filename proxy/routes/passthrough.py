"""
Pass-through endpoints: health check, models list, and catch-all.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from proxy import config
from proxy.routes._helpers import upstream_headers, upstream_url

logger = logging.getLogger("proxy")

router = APIRouter()


# ── Health check ────────────────────────────────────────────────────


@router.get("/health")
async def health(request: Request) -> dict:
    key_manager = request.app.state.key_manager

    result: dict = {
        "status": "ok",
        "upstream": config.UPSTREAM_BASE_URL,
        "features": {
            "response_validator": config.ENABLE_RESPONSE_VALIDATOR,
            "tool_fixer": config.ENABLE_TOOL_FIXER,
            "request_logging": config.ENABLE_REQUEST_LOGGING,
            "key_rotation": key_manager is not None,
        },
    }
    if key_manager is not None:
        result["keys"] = {
            "total": key_manager.key_count,
            "active": key_manager.active_key_count,
        }
    return result


# ── Models endpoint (pass-through) ─────────────────────────────────


@router.get("/v1/models")
@router.get("/models")
async def list_models(request: Request) -> Response:
    http_client: httpx.AsyncClient = request.app.state.http_client
    headers = upstream_headers(request)

    try:
        resp = await http_client.get(upstream_url("/models"), headers=headers)
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


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str) -> Response:
    """Forward any other /v1/* request to upstream as-is."""
    http_client: httpx.AsyncClient = request.app.state.http_client
    headers = upstream_headers(request)

    method = request.method.upper()
    url = upstream_url(f"/{path}")

    try:
        if method in ("POST", "PUT", "PATCH"):
            body = await request.body()
            resp = await http_client.request(method, url, content=body, headers=headers)
        else:
            resp = await http_client.request(method, url, headers=headers)

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
