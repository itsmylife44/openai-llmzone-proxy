# LLMZone OpenAI Proxy

A FastAPI proxy server that sits between OpenAI-compatible clients and [LLMZone](https://api.llmzone.net/v1). It fixes non-conformant responses from LLMZone so that standard OpenAI API clients work correctly.

## Overview

LLMZone's API is largely OpenAI-compatible but contains several response formatting bugs. This proxy intercepts requests and responses, repairs the issues on the fly, and passes conformant responses back to the client — with no changes required on the client side.

### Bugs Fixed

| # | Issue | Fix |
|---|-------|-----|
| 1 | `finish_reason: "stop"` when `tool_calls` are present (non-streaming) | Changed to `"tool_calls"` |
| 2 | `finish_reason: "tool_calls"` when no `tool_calls` present (streaming) | Changed to `"stop"` |
| 3 | `id` field uses `resp_*` prefix instead of `chatcmpl-*` | Rewritten to `chatcmpl-*` |
| 4 | Extra `native_finish_reason` field in response | Removed |
| 5 | Extra `reasoning_content` field in response | Removed |
| 6 | `tool_calls: null` present instead of absent | Removed |
| 7 | Streaming `tool_calls[].index` starts at `1` instead of `0` | Re-mapped to 0-based |

## Features

- **Tool-call fixer** — 7 targeted repairs applied to every response
- **5-stage response validator** — validates responses against the OpenAI API spec before forwarding
- **Chat Completions API** (`/v1/chat/completions`) — primary proxy endpoint
- **Responses API** (`/v1/responses`) — bidirectional conversion to/from Chat Completions
- **SSE streaming** — fixes applied on-the-fly during streaming responses
- **API key rotation** — round-robin selection with 2-hour cooldown blacklisting on errors
- **Request/response logging** — JSON logs with before/after diffs (toggleable)
- **Fully configurable** — all features toggled via environment variables

## Requirements

- Docker and Docker Compose, or Python 3.12+

## Quick Start

```bash
git clone <repo-url>
cd proxy-test

cp .env.example .env
# Edit .env and fill in your UPSTREAM_API_KEYS

docker compose up -d
```

The proxy listens on port `8081` by default.

Point your OpenAI-compatible client at `http://localhost:8081` and use any API key value (the proxy substitutes its own upstream keys).

## Configuration

Copy `.env.example` to `.env` and set the following variables:

```env
UPSTREAM_BASE_URL=https://api.llmzone.net/v1
UPSTREAM_TIMEOUT=300
UPSTREAM_API_KEYS=sk-your-key-1,sk-your-key-2

KEY_COOLDOWN_SECONDS=7200

PROXY_HOST=0.0.0.0
PROXY_PORT=8081

ENABLE_TOOL_FIXER=true
ENABLE_RESPONSE_VALIDATOR=true
ENABLE_REQUEST_LOGGING=false

LOG_DIR=./logs
LOG_LEVEL=WARNING
```

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | `https://api.llmzone.net/v1` | Upstream API base URL |
| `UPSTREAM_TIMEOUT` | `300` | Request timeout in seconds |
| `UPSTREAM_API_KEYS` | — | Comma-separated list of API keys |
| `KEY_COOLDOWN_SECONDS` | `7200` | Blacklist duration after key error (seconds) |
| `PROXY_HOST` | `0.0.0.0` | Bind address |
| `PROXY_PORT` | `8081` | Listen port |
| `ENABLE_TOOL_FIXER` | `true` | Enable tool-call response fixes |
| `ENABLE_RESPONSE_VALIDATOR` | `true` | Enable 5-stage OpenAI spec validation |
| `ENABLE_REQUEST_LOGGING` | `false` | Log full request/response JSON with diffs |
| `LOG_DIR` | `./logs` | Directory for log files |
| `LOG_LEVEL` | `WARNING` | Python logging level |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat Completions — primary proxy endpoint |
| `POST` | `/v1/responses` | Responses API — converted to Chat Completions internally |
| `GET` | `/v1/models` | Models list — pass-through |
| `GET` | `/health` | Health check |
| `*` | `/v1/*` | Catch-all pass-through for all other endpoints |

## Project Structure

```
├── proxy/                         # Python package
│   ├── app.py                     # FastAPI app factory, lifespan, entrypoint
│   ├── config.py                  # Settings loaded from environment
│   ├── core/                      # Domain logic (no FastAPI dependency)
│   │   ├── key_manager.py         # API key rotation and blacklisting
│   │   ├── request_log.py         # JSON request/response logger
│   │   ├── response_validator.py  # 5-stage OpenAI spec validator
│   │   ├── responses_converter.py # Responses API <-> Chat Completions converter
│   │   └── tool_call_fixer.py     # Tool-call response repairs
│   ├── handlers/                  # Request handling logic
│   │   ├── non_streaming.py       # Non-streaming request handlers
│   │   └── streaming.py           # SSE streaming handlers
│   └── routes/                    # API route definitions
│       ├── _helpers.py            # Shared route utilities
│       ├── chat.py                # POST /v1/chat/completions
│       ├── passthrough.py         # Health, models, catch-all
│       └── responses.py           # POST /v1/responses
├── tests/                         # Test suite
├── Dockerfile                     # Multi-stage build, Python 3.12-slim
├── docker-compose.yml             # Container orchestration
├── .env.example                   # Configuration template
├── pyproject.toml                 # Project metadata and tooling
└── requirements.txt               # Python dependencies
```

## Running Without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env

uvicorn proxy.app:app --host 0.0.0.0 --port 8081
```

## Notes

- **Never commit real API keys.** The `.env` file in the repository contains placeholder values only. Add `.env` to `.gitignore`.
- For production, set `LOG_LEVEL=WARNING` and `ENABLE_REQUEST_LOGGING=false` to avoid writing sensitive request data to disk.
- The Docker container does not expose ports externally by default — it is designed for Docker internal networking. Adjust `docker-compose.yml` if you need external access.

## License

MIT
