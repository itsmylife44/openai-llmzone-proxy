"""
OpenAI-compatible proxy server — application factory.

Creates the FastAPI app with lifespan management, registers all routers,
and provides a main() entrypoint for running with uvicorn.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from proxy import config
from proxy.core.key_manager import KeyManager
from proxy.core.request_log import setup_logging
from proxy.routes.chat import router as chat_router
from proxy.routes.passthrough import router as passthrough_router
from proxy.routes.responses import router as responses_router

logger = logging.getLogger("proxy")


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Logging ─────────────────────────────────────────────────────
    setup_logging(config.LOG_LEVEL)

    # ── Key Manager (rotation + blacklisting) ──────────────────────
    key_manager: KeyManager | None = None
    if config.UPSTREAM_API_KEYS:
        key_manager = KeyManager(
            keys=config.UPSTREAM_API_KEYS,
            cooldown_seconds=config.KEY_COOLDOWN_SECONDS,
        )
        logger.info(
            "Key rotation enabled: %d key(s), cooldown=%ds",
            key_manager.key_count,
            config.KEY_COOLDOWN_SECONDS,
        )
    else:
        logger.info("Key rotation disabled: no API keys configured")

    # ── HTTP client ─────────────────────────────────────────────────
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.UPSTREAM_TIMEOUT, connect=30.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
        follow_redirects=True,
    )
    app.state.key_manager = key_manager
    logger.info("HTTP client initialized")
    yield
    if not app.state.http_client.is_closed:
        await app.state.http_client.aclose()
        logger.info("HTTP client closed")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLMZone OpenAI Proxy",
    description="OpenAI-compatible proxy with response validation & tool-call fixing",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Routers ──────────────────────────────────────────────────────────

app.include_router(chat_router)
app.include_router(responses_router)
app.include_router(passthrough_router)


# ── Entrypoint ───────────────────────────────────────────────────────


def main() -> None:
    import uvicorn

    uvicorn.run(
        "proxy.app:app",
        host=config.PROXY_HOST,
        port=config.PROXY_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
