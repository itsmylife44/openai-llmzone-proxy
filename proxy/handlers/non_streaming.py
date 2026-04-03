"""
Non-streaming handler functions for the proxy.

Handles both Chat Completions and Responses API non-streaming requests.
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy

import httpx
from fastapi.responses import JSONResponse

from proxy import config
from proxy.core.key_manager import KeyManager
from proxy.core.request_log import log_response
from proxy.core.response_validator import validate_response
from proxy.core.responses_converter import chat_to_responses_response
from proxy.core.tool_call_fixer import fix_tool_calls_response

logger = logging.getLogger("proxy")


# ── Shared helpers ──────────────────────────────────────────────────


def apply_fixes(response_data: dict) -> list[str]:
    """Run the full validation + fix pipeline on a response dict."""
    all_fixes: list[str] = []
    if config.ENABLE_RESPONSE_VALIDATOR:
        all_fixes.extend(validate_response(response_data))
    if config.ENABLE_TOOL_FIXER:
        all_fixes.extend(fix_tool_calls_response(response_data))
    return all_fixes


def report_upstream_error(
    key_manager: KeyManager | None,
    api_key: str | None,
    status_code: int,
    error_body: str = "",
) -> None:
    if key_manager is not None and api_key is not None:
        key_manager.report_error(key=api_key, status_code=status_code, error_body=error_body)


def report_upstream_success(key_manager: KeyManager | None, api_key: str | None) -> None:
    if key_manager is not None and api_key is not None:
        key_manager.report_success(api_key)


# ── Non-streaming handlers ──────────────────────────────────────────


async def handle_non_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    body: dict,
    t0: float,
    http_client: httpx.AsyncClient,
    key_manager: KeyManager | None,
    api_key: str | None = None,
) -> JSONResponse:
    resp = await http_client.post(upstream_url, json=body, headers=headers)
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

    # Guard against JSON null body
    if upstream_data is None:
        upstream_data = {}

    # ── If upstream returned an error, report to key manager & pass through
    if resp.status_code >= 400:
        error_text = json.dumps(upstream_data) if upstream_data else ""
        report_upstream_error(key_manager, api_key, resp.status_code, error_text)
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
    report_upstream_success(key_manager, api_key)

    # ── Validate & fix ──────────────────────────────────────────────
    upstream_raw = deepcopy(upstream_data)
    fixes = apply_fixes(upstream_data)

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


async def handle_responses_non_streaming(
    request_id: str,
    upstream_url: str,
    headers: dict[str, str],
    chat_body: dict,
    original_body: dict,
    t0: float,
    http_client: httpx.AsyncClient,
    key_manager: KeyManager | None,
    api_key: str | None = None,
) -> JSONResponse:
    """Send as Chat Completions, convert response back to Responses API format."""
    resp = await http_client.post(upstream_url, json=chat_body, headers=headers)
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

    # Guard against JSON null body
    if upstream_data is None:
        upstream_data = {}

    if resp.status_code >= 400:
        error_text = json.dumps(upstream_data) if upstream_data else ""
        report_upstream_error(key_manager, api_key, resp.status_code, error_text)
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
    report_upstream_success(key_manager, api_key)

    # ── Validate & fix the Chat Completions response ────────────────
    upstream_raw = deepcopy(upstream_data)
    fixes = apply_fixes(upstream_data)

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
