# Release Notes — v1.1.0

**Date:** 2026-03-19
**Type:** Minor release (new capabilities, no breaking changes)

## Summary

Geometric Brain MCP goes from working prototype to hardened, configurable, observable service. All existing public APIs remain unchanged. New behavior is opt-in via environment variables.

## Highlights

### Production-ready configuration

All runtime settings now flow through environment variables with sane defaults. Local development requires zero configuration. Production deployments control auth, rate limits, CORS, logging, and input caps entirely through `GB_*` env vars. See `.env.example` for the full list.

### API key authentication

Optional Bearer token auth protects all non-public endpoints. Enable with `GB_AUTH_ENABLED=true` and provide keys via `GB_API_KEYS`. Supports multiple keys for rotation. Public paths (`/healthz`, `/readyz`, `/docs`, `/openapi.json`) remain open.

### Rate limiting and request guardrails

Per-IP rate limiting (configurable RPM), pre-parse body size rejection, and validated input caps on text length, hidden-state dimensions, and eigenvalue counts. All limits visible at `GET /v1/meta/capabilities`.

### Structured logging and request tracing

Every request gets an `X-Request-ID` (auto-generated or client-provided) that appears in the response header, response body, and all log lines. All logs are structured JSON. Startup banner shows deployment config.

### Standardized error payloads

All errors (401, 413, 422, 429, 500) follow a consistent shape: `{status, error_code, detail, request_id, schema_version}`.

### CI pipeline

GitHub Actions runs on every push/PR to main: compile check, import verification, ruff lint, ruff format, and full pytest suite on Python 3.11 and 3.12.

### Honest validation posture

New README sections document exactly what has been validated (math correctness, test coverage) and what has not (no hallucination detection claim, no committed benchmark results). Evidence Tiers table ranks endpoints by analytical strength.

### Test expansion

138 tests (up from 90), covering auth, rate limiting, body size, input guardrails, request tracing, error payloads, config toggle wiring, and version visibility.

## Upgrade path

This is a **non-breaking** minor release. Existing integrations continue to work with no changes.

**If you use the REST API:** all existing endpoints, request shapes, and response shapes are unchanged. New middleware is disabled by default.

**If you use the MCP server:** no changes to tool signatures or behavior.

**If you import `spectral_engine` directly:** no public API changes.

To enable new features, set environment variables per the configuration table in README.md.

## Version

| Source | Value |
| -------- | ------- |
| `config.SCHEMA_VERSION` | `1.1.0` |
| `pyproject.toml [project] version` | `1.1.0` |
| Git tag | `v1.1.0` |
