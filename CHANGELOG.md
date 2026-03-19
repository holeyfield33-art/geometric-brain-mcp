# Changelog

All notable changes to Geometric Brain MCP are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-03-19

### Added
- **Configuration layer** (`config.py`): all runtime settings driven by environment variables with sane defaults. Zero config required for local dev.
- **API key authentication**: optional Bearer token auth on all non-public endpoints. Controlled by `GB_AUTH_ENABLED` and `GB_API_KEYS`.
- **Rate limiting**: optional per-IP request throttling. Controlled by `GB_RATE_LIMIT_ENABLED` and `GB_RATE_LIMIT_RPM`.
- **Body size limit**: pre-parse rejection of oversized payloads via `GB_MAX_BODY_BYTES` (default 10 MB).
- **Input guardrails**: configurable caps on text length, hidden-state dimensions/samples, and eigenvalue count. Enforced via Pydantic validation.
- **Structured logging**: JSON log lines for all requests and errors via Python `logging`. Startup banner with deployment config.
- **Request tracing**: `X-Request-ID` header on every response; echoed if client provides one. ID appears in response body and all log lines.
- **Standardized error payloads**: all error responses follow `{status, error_code, detail, request_id, schema_version}` shape.
- **CI pipeline** (`.github/workflows/ci.yml`): Python 3.11 + 3.12 matrix with compile check, import check, ruff lint, ruff format, pytest.
- **Linting config** (`pyproject.toml`): ruff with E, F, W, I, B rules; line-length 120.
- **Lightweight CI dependencies** (`requirements-ci.txt`): runtime + test deps without GPU/ML packages.
- **Environment template** (`.env.example`): documents all `GB_*` variables.
- **48 new tests**: auth (13), rate limiting (5), body size (3), input guardrails (5), request tracing (6), standardized errors (5), config toggles (6), version visibility (5). Total: 138.
- **Validation Status section** in README: honest documentation of what is and is not validated.
- **Evidence Tiers table** in README: ranks endpoints by analytical strength (Tier 1 Direct → Tier 3 Proxy).
- **Release checklist** in PACKAGE_DELIVERY.md §9.
- **PyPI packaging** (`[build-system]` in `pyproject.toml`): sdist and wheel build via `python -m build`. Runtime dependencies declared in `[project.dependencies]`. Optional extras: `[bridge]` for GPU/ML deps, `[dev]` for test/lint tools.
- **Launch artifacts**: CHANGELOG.md, RELEASE_NOTES.md, DEFERRED_WORK.md, MVP_LAUNCH_CHECKLIST.md.

### Changed
- `api.py`: middleware chain (request_id → logging → body_size → rate_limit → auth → handler). All endpoint ValueError catches use `raise ... from None` (B904).
- `server.py`: Pydantic schemas now read limits from `config.*`. SCHEMA_VERSION sourced from config.
- `spectral_engine.py`: removed unused variable in `manifold_audit()` (F841).
- `bridge_validation.py`: docstring rewritten — marked as research script, no committed results, interpretation caveats.
- `README.md`: comprehensive rewrite — added configuration, authentication, rate limiting, logging, deployment, security notes, validation status, evidence tiers, operational reference.
- `PACKAGE_DELIVERY.md`: field names corrected, §7 rewritten for bridge validation clarity, §9 rewritten as pre-release checklist.
- `test_e2e.py`: import ordering fixed (I001), empty f-string fixed (F541).

### Fixed
- B904: `raise HTTPException` in `except ValueError` blocks now uses `from None`.
- F841: unused `computed_eigenvalues` variable removed from `spectral_engine.py`.
- F541: empty f-string in test fixture replaced with plain string.
- I001: import ordering in test file corrected.

## [1.0.2] — 2026-03-18

- Diagnostics and typing issue fixes.

## [1.0.1] — 2026-03-18

- Professional README and delivery runbook.

## [1.0.0] — 2026-03-18

- Initial release: spectral engine, MCP server, REST API, bridge validation script.
