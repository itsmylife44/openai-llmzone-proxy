"""Proxy configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


# ── Upstream (LLMZone) ──────────────────────────────────────────────
UPSTREAM_BASE_URL: str = _env("UPSTREAM_BASE_URL", "https://api.llmzone.net/v1")
UPSTREAM_API_KEY: str = _env("UPSTREAM_API_KEY", "")
UPSTREAM_TIMEOUT: int = _env_int("UPSTREAM_TIMEOUT", 300)

# ── Key Rotation ────────────────────────────────────────────────────
# Comma-separated list of API keys for rotation.
# Falls back to UPSTREAM_API_KEY if not set.
_raw_keys = _env("UPSTREAM_API_KEYS", "")
UPSTREAM_API_KEYS: list[str] = (
    [k.strip() for k in _raw_keys.split(",") if k.strip()]
    if _raw_keys.strip()
    else ([UPSTREAM_API_KEY] if UPSTREAM_API_KEY else [])
)

# Cooldown in seconds when a key is blacklisted (default: 2 hours)
KEY_COOLDOWN_SECONDS: int = _env_int("KEY_COOLDOWN_SECONDS", 7200)

# Enable/disable key rotation (auto-enabled when multiple keys present)
ENABLE_KEY_ROTATION: bool = _env_bool("ENABLE_KEY_ROTATION", len(UPSTREAM_API_KEYS) > 1)

# ── Proxy server ────────────────────────────────────────────────────
PROXY_HOST: str = _env("PROXY_HOST", "0.0.0.0")
PROXY_PORT: int = _env_int("PROXY_PORT", 8080)

# ── Feature flags ───────────────────────────────────────────────────
ENABLE_TOOL_FIXER: bool = _env_bool("ENABLE_TOOL_FIXER", True)
ENABLE_RESPONSE_VALIDATOR: bool = _env_bool("ENABLE_RESPONSE_VALIDATOR", True)
ENABLE_REQUEST_LOGGING: bool = _env_bool("ENABLE_REQUEST_LOGGING", True)

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR: Path = Path(_env("LOG_DIR", "./logs"))
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
