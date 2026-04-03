from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from proxy import config
from proxy.core.key_manager import KeyManager


def upstream_url(path: str) -> str:
    """Build the full upstream URL for a given path."""
    base = config.UPSTREAM_BASE_URL.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


def get_api_key(key_manager: KeyManager | None) -> str | None:
    """Get the next available API key from the key manager.

    Returns the key string, or None if all keys are blacklisted.
    """
    if key_manager is not None:
        return key_manager.get_key()
    # Fallback: static key from config
    return config.UPSTREAM_API_KEY or None


def upstream_headers(request: Request, api_key: str | None = None) -> dict[str, str]:
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


def all_keys_blacklisted_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "All upstream API keys are currently rate-limited. Please try again later.",
                "type": "rate_limit_error",
            }
        },
    )
