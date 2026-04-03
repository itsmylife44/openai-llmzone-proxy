"""
JSON-based request/response logger for the OpenAI proxy.

Logs every request and response as structured JSON files for debugging.
Shows diffs between original upstream response and validated/fixed response.
"""

import copy
import json
import logging
import time
import uuid
from pathlib import Path

import config
from config import LOG_DIR, LOG_LEVEL

logger = logging.getLogger("proxy.logging")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure Python logging with structured console output."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.DEBUG), format=fmt)
    logging.getLogger("proxy").setLevel(getattr(logging, log_level.upper(), logging.DEBUG))


def _sanitize_headers(headers: dict) -> dict:
    """Mask Authorization header value, keeping first 10 chars."""
    sanitized = copy.copy(headers)
    for key in list(sanitized.keys()):
        if key.lower() == "authorization":
            val = sanitized[key]
            sanitized[key] = (val[:10] + "...") if len(val) > 10 else val
    return sanitized


def _truncate(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def _write_json_file(path: Path, data: dict) -> None:
    """Write JSON to a file; silently ignore failures."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to write log file %s: %s", path, exc)


def log_request(
    request_id: str,
    method: str,
    path: str,
    headers: dict,
    body: dict | None,
) -> None:
    """Log an incoming request to both console and file."""
    model = body.get("model") if body else None
    stream = body.get("stream") if body else None
    has_tools = bool(body.get("tools")) if body else False

    logger.info(
        "[REQ %s] %s %s model=%s stream=%s has_tools=%s",
        request_id,
        method,
        path,
        model,
        stream,
        has_tools,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = LOG_DIR / f"{timestamp}_{request_id}_req.json"

    _write_json_file(
        file_path,
        {
            "request_id": request_id,
            "method": method,
            "path": path,
            "headers": _sanitize_headers(headers),
            "body": body,
        },
    )


def log_response(
    request_id: str,
    status_code: int,
    upstream_raw: dict | None,
    validated: dict | None,
    fixes: list[str],
    duration_ms: float,
) -> None:
    """Log a response with before/after comparison."""
    # Extract finish_reason and has_tool_calls from validated or upstream
    response = validated or upstream_raw
    finish_reason = None
    has_tool_calls = False
    if response:
        choices = response.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason")
            msg = choices[0].get("message") or {}
            has_tool_calls = bool(msg.get("tool_calls"))

    logger.info(
        "[RES %s] status=%s duration=%dms fixes=%d finish_reason=%s has_tool_calls=%s",
        request_id,
        status_code,
        int(duration_ms),
        len(fixes),
        finish_reason,
        has_tool_calls,
    )

    for fix in fixes:
        logger.info("[RES %s] fix applied: %s", request_id, fix)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = LOG_DIR / f"{timestamp}_{request_id}_res.json"

    _write_json_file(
        file_path,
        {
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "fixes_applied": fixes,
            "upstream_raw": upstream_raw,
            "validated_response": validated,
        },
    )


def log_streaming_summary(
    request_id: str,
    chunk_count: int,
    fixes: list[str],
    duration_ms: float,
    had_tool_calls: bool,
    final_finish_reason: str | None,
) -> None:
    """Log summary for a streaming response."""
    logger.info(
        "[STREAM %s] chunks=%d duration=%dms tool_calls=%s finish_reason=%s fixes=%d",
        request_id,
        chunk_count,
        int(duration_ms),
        had_tool_calls,
        final_finish_reason,
        len(fixes),
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = LOG_DIR / f"{timestamp}_{request_id}_stream.json"

    _write_json_file(
        file_path,
        {
            "request_id": request_id,
            "chunk_count": chunk_count,
            "duration_ms": duration_ms,
            "had_tool_calls": had_tool_calls,
            "final_finish_reason": final_finish_reason,
            "fixes_applied": fixes,
        },
    )


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"
