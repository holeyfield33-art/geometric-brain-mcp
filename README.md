# Geometric Brain MCP

![CI](https://github.com/holeyfield33-art/geometric-brain-mcp/actions/workflows/ci.yml/badge.svg)

Lightweight spectral diagnostics engine for AI systems. Analyze text proxies, hidden states, or eigenvalue spectra to estimate model health, classify spectral regime, compare runs or models, and generate correction guidance.

Accessible via **MCP tools**, **REST API**, or **direct Python import**.

## Current Scope

### What it does

- Estimates eigenvalue spacing ratio (`<r>`) from text, hidden states, or raw eigenvalue spectra
- Classifies spectral regime: GUE-like (rigid/coherent), Poisson-like (uncorrelated), or intermediate
- Computes a spectral health score (0–100) and confidence estimate
- Generates bounded correction signals with direction and magnitude
- Compares two models/checkpoints/layers side-by-side on spectral health

### What it does not do

- It does not guarantee detection of hallucination or factual errors
- It does not access model internals at runtime (you provide the data)
- It does not serve as a complete observability platform, dashboard, or alerting system
- It does not fine-tune, train, or modify models
- It does not provide safety guarantees

### Strongest use cases

- Monitoring spectral drift across model checkpoints or prompt strategies
- Comparing spectral health between two models on the same data
- Lightweight observability signal in agent or pipeline toolchains
- Research exploration of spectral properties in transformer hidden states

### Limitations

- **Text proxy mode** is an indirect estimate — it does not access true eigenvalue structure. Use it as a coarse signal, not a precise measurement.
- Confidence depends on sample size. Small inputs produce low-confidence results.
- Regime classification thresholds are fixed heuristics, not empirically tuned per-model. Interpretation should account for your model and data context.
- The spectral health score is a derived metric, not a ground-truth measure of model quality.
- No published validation results are shipped with this repository. The bridge validation script exists as a harness but has not produced committed results (see [Validation Status](#validation-status) below).

## Validation Status

This project ships **diagnostic tooling**, not validated detection claims. The distinction matters.

### What has been validated

- The engine correctly computes eigenvalue spacing ratios and classifies regimes against known GUE/Poisson reference values.
- All four public functions are covered by 138 automated tests exercising happy paths, edge cases, error handling, and API integration.
- The mathematical operations (Gram matrix, eigendecomposition, spacing ratio, Marchenko-Pastur comparison) are standard linear algebra — they compute what they claim to compute.

### What has NOT been validated

- **No hallucination detection claim is made.** The engine measures spectral properties; whether those properties predict truthfulness in a given model is an open research question.
- **No benchmark results are committed to this repository.** The `bridge_validation.py` script is a research harness (see below), but its output (`hidden_state_validation.json`) has not been run and checked in.
- **Text proxy mode has not been shown to correlate with model behavior.** An earlier experiment (noted in `bridge_validation.py`) reported AUROC = 0.567 on TruthfulQA text proxy — effectively chance.

### bridge_validation.py

`bridge_validation.py` is a **research script**, not a runtime component. It:

1. Loads a HuggingFace causal LM (TinyLlama or Gemma)
2. Extracts hidden states on TruthfulQA (truthful vs. incorrect answers)
3. Runs `manifold_audit()` on each sample
4. Computes AUROC for separating truthful from hallucinated responses
5. Saves results to `hidden_state_validation.json`

**Requirements:** GPU recommended, plus `torch`, `transformers`, `datasets`, `scikit-learn` (not included in `requirements-ci.txt`).

**Status:** The script is functional and tested for import correctness, but no committed results exist. If you run it and get strong separation, that is evidence for your model on your data — not a general claim.

### Interpreting results

All outputs — `r_ratio`, `spectral_health_score`, `regime`, `confidence` — describe **spectral geometry**, not model correctness. A "warning" status means the spacing ratio diverges from GUE-like rigidity; it does not mean the model is hallucinating. Users should establish their own baselines and interpret changes relative to those baselines.

## Evidence Tiers

Not all endpoints carry equal analytical weight. The table below ranks them by how directly they access spectral structure.

| Tier | Endpoint | Input | What it measures | Strength |
| ------ | ---------- | ------- | ----------------- | ---------- |
| 1 — Direct | `manifold-audit` (hidden_states) | Model hidden states | True eigenvalue spacing from Gram matrix | Strongest — operates on actual latent geometry |
| 1 — Direct | `manifold-audit` (eigenvalues) | Precomputed eigenvalues | Spacing ratio and regime from real spectrum | Strongest — skips Gram step, same analysis |
| 2 — Derived | `compare-models` | Two sets of hidden states or eigenvalues | Side-by-side tier-1 audit | Strong — composes two direct analyses |
| 3 — Proxy | `health-check` | Raw text | Character-level token spacing as eigenvalue proxy | Weakest — no model access, indirect estimate |
| Utility | `compute-correction` | Scalar r_ratio | Deterministic correction signal | N/A — pure function, no analysis |

Tier 1 and 2 endpoints are the scientifically meaningful surface. Tier 3 is a convenience proxy. Utility endpoints perform no spectral analysis.

## Analysis Modes

### Text proxy mode (Tier 3 — Proxy)

Encodes raw text as character-level tokens, applies sliding-window analysis, and estimates eigenvalue spacing ratio from token-level differences. This is the lightest mode — no model access required. **This is an indirect proxy, not a direct spectral measurement.**

**Function:** `spectral_health_check(text, window_size=128, stride=64)`
**Endpoint:** `POST /v1/brain/health-check`

**Best for:** quick screening, agent toolchains, situations where hidden states are unavailable. Not suitable for rigorous analysis.
**Caveat:** proxy measurement only. Results are approximate.

### Hidden-state mode (Tier 1 — Direct)

Accepts a 2D matrix of hidden-state vectors `[tokens × dimensions]`, computes the Gram matrix, extracts eigenvalues via `eigh`, and performs full spectral analysis.

**Function:** `manifold_audit(hidden_states=[[...], ...], normalize=True, center=True)`
**Endpoint:** `POST /v1/brain/manifold-audit` with `"source_type": "hidden_states"`

**Best for:** direct model diagnostics when you can extract hidden states from inference. This is the strongest analytical mode.

### Eigenvalue mode (Tier 1 — Direct)

Accepts a precomputed list of eigenvalues and performs spectral analysis directly. Skips Gram matrix computation.

**Function:** `manifold_audit(eigenvalues=[...])`
**Endpoint:** `POST /v1/brain/manifold-audit` with `"source_type": "eigenvalues"`

**Best for:** research workflows, pre-processed spectra, custom eigenvalue pipelines.

## Interface Options

| Interface | Best for | Entry point |
| --- | --- | --- |
| MCP Server | MCP-compatible agents and orchestration runtimes | `python server.py` |
| HTTP API | Service-to-service and language-agnostic clients | `python api.py` |
| Python Module | Direct embedding in pipelines and research code | `import spectral_engine` |

## Tools / Endpoints

### `brain_health_check`

Text proxy analysis. Returns `status`, `r_ratio`, `shi_score`, `regime`, `drift_warning`, `confidence`, `windows_analyzed`, `spacing_count`.

### `brain_manifold_audit`

Full spectral audit from hidden states or eigenvalues. Returns `spectral_health_score`, `mean_r_ratio`, `variance_r_ratio`, `spectral_regime`, `lambda_2`, `zeta_score`, `gue_distance`, `poisson_distance`, `confidence`, `summary`.

### `brain_compute_correction`

Correction signal from current spacing ratio to target. Returns `delta`, `intervention_signal`, `magnitude`, `direction`, `recommended_action`, `recommended_sigma`, `confidence`.

### `brain_compare_models`

Side-by-side comparison. Returns `healthier_model`, `delta_health_score`, `left_health_score`, `right_health_score`, `left_regime`, `right_regime`, `comparative_summary`, `warnings`.

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes heavy research dependencies (torch, transformers). For API/MCP runtime only, you need: `numpy`, `pydantic`, `mcp`, `fastapi`, `uvicorn`, `httpx`.

### Start MCP server (stdio)

```bash
python server.py
```

### Start MCP server (HTTP mode)

```bash
python server.py --http --host 0.0.0.0 --port 8000
```

### Start REST API

```bash
python api.py
```

### Python library usage

```python
from spectral_engine import spectral_health_check, manifold_audit, compute_correction, compare_models

# Text proxy mode
result = spectral_health_check("Example text for spectral analysis.")
print(result["status"], result["r_ratio"], result["regime"])

# Eigenvalue mode
audit = manifold_audit(eigenvalues=[0.9, 1.1, 1.4, 1.8, 2.1])
print(audit["status"], audit["spectral_health_score"])

# Correction signal
corr = compute_correction(current_r_ratio=0.52)
print(corr["direction"], corr["delta"])

# Model comparison
cmp = compare_models(
    left_eigenvalues=[1.0, 1.2, 1.5, 1.7, 2.0],
    right_eigenvalues=[0.9, 1.1, 1.3, 1.6, 1.9],
)
print(cmp["healthier_model"], cmp["delta_health_score"])
```

### Example REST calls

```bash
# Health check (text proxy)
curl -s -X POST http://localhost:8000/v1/brain/health-check \
  -H "Content-Type: application/json" \
  -d '{"text": "The model response appears stable and consistent."}'

# Manifold audit (eigenvalues)
curl -s -X POST http://localhost:8000/v1/brain/manifold-audit \
  -H "Content-Type: application/json" \
  -d '{"source_type": "eigenvalues", "eigenvalues": [0.9, 1.1, 1.3, 1.6, 2.0]}'

# Compute correction
curl -s -X POST http://localhost:8000/v1/brain/compute-correction \
  -H "Content-Type: application/json" \
  -d '{"current_r_ratio": 0.51}'

# Compare models
curl -s -X POST http://localhost:8000/v1/brain/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "left": {"model_label": "model_a", "source_type": "eigenvalues", "eigenvalues": [0.8, 1.0, 1.2, 1.5, 1.7]},
    "right": {"model_label": "model_b", "source_type": "eigenvalues", "eigenvalues": [0.9, 1.2, 1.4, 1.6, 1.9]}
  }'
```

## Constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `GUE_R` | 0.578 | Target spacing ratio — GUE-like spectral rigidity |
| `POISSON_R` | 0.386 | Baseline spacing ratio — uncorrelated/Poisson-like |

Interpretation depends on your model, data, and operating context. These constants define classification thresholds, not guarantees.

## Configuration

All runtime settings are centralized in `config.py` and driven by environment variables. No configuration is required for local development — defaults are sane.

Copy the example file to get started:

```bash
cp .env.example .env
```

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `GB_HOST` | `0.0.0.0` | Bind address |
| `GB_PORT` | `8000` | Listen port (also reads `PORT` as fallback) |
| `GB_AUTH_ENABLED` | `false` | Enable API key authentication |
| `GB_API_KEYS` | *(empty)* | Comma-separated accepted API keys |
| `GB_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `GB_RATE_LIMIT_ENABLED` | `false` | Enable per-IP rate limiting |
| `GB_RATE_LIMIT_RPM` | `60` | Max requests per minute per IP |
| `GB_LOG_LEVEL` | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR |
| `GB_ENVIRONMENT` | `development` | Environment label (development, staging, production) |
| `GB_MAX_TEXT_LENGTH` | `20000` | Max text input characters |
| `GB_MAX_HIDDEN_STATE_SAMPLES` | `8192` | Max hidden-state sample rows |
| `GB_MAX_HIDDEN_STATE_DIMS` | `4096` | Max hidden-state dimensions |
| `GB_MAX_EIGENVALUES` | `65536` | Max eigenvalue list length |
| `GB_MAX_BODY_BYTES` | `10485760` | Max request body size in bytes (10 MB) |

### Production example

```bash
GB_AUTH_ENABLED=true
GB_API_KEYS=sk-prod-key-1,sk-prod-key-2
GB_CORS_ORIGINS=https://app.example.com,https://admin.example.com
GB_RATE_LIMIT_ENABLED=true
GB_RATE_LIMIT_RPM=30
GB_LOG_LEVEL=WARNING
GB_ENVIRONMENT=production
GB_PORT=10000
```

## Authentication

API key authentication is available for the REST API. Disabled by default for local development.

### Enabling auth

```bash
GB_AUTH_ENABLED=true
GB_API_KEYS=sk-your-key-here
```

When enabled, all endpoints except `/healthz`, `/readyz`, `/docs`, and `/openapi.json` require a valid API key.

### Sending the key

Include it as a Bearer token:

```bash
curl -s -X POST http://localhost:8000/v1/brain/health-check \
  -H "Authorization: Bearer sk-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Example text for analysis."}'
```

### Failure response

Missing or invalid keys return `401`:

```json
{
  "status": "error",
  "error_code": "UNAUTHORIZED",
  "detail": "Invalid or missing API key. Provide a valid key via the Authorization: Bearer <key> header."
}
```

### Multiple keys

Provide a comma-separated list to support key rotation:

```bash
GB_API_KEYS=sk-primary-key,sk-secondary-key
```

### Local development

With `GB_AUTH_ENABLED=false` (the default), no authentication is required. All endpoints are open.

## Rate Limiting & Guardrails

### Rate limiting

In-process per-IP rate limiting. Disabled by default.

```bash
GB_RATE_LIMIT_ENABLED=true
GB_RATE_LIMIT_RPM=60
```

When a client exceeds the limit, the API returns `429`:

```json
{
  "status": "error",
  "error_code": "RATE_LIMITED",
  "detail": "Rate limit exceeded. Max 60 requests per minute.",
  "request_id": "...",
  "schema_version": "1.0.0"
}
```

Public paths (`/healthz`, `/readyz`, `/docs`, `/openapi.json`) are exempt.

### Body size limit

Requests larger than `GB_MAX_BODY_BYTES` (default 10 MB) are rejected before JSON parsing with `413`:

```json
{
  "status": "error",
  "error_code": "PAYLOAD_TOO_LARGE",
  "detail": "Request body exceeds maximum size of 10485760 bytes.",
  "request_id": "...",
  "schema_version": "1.0.0"
}
```

### Input caps

All caps are configurable via environment variables and enforced by Pydantic validation (422 on violation):

| Limit | Default | Env var |
| --- | --- | --- |
| Text input length | 20 000 chars | `GB_MAX_TEXT_LENGTH` |
| Hidden-state samples | 8 192 rows | `GB_MAX_HIDDEN_STATE_SAMPLES` |
| Hidden-state dimensions | 4 096 cols | `GB_MAX_HIDDEN_STATE_DIMS` |
| Eigenvalue count | 65 536 | `GB_MAX_EIGENVALUES` |

Current limits are always visible at `GET /v1/meta/capabilities`.

## Testing

```bash
python -m pytest test_e2e.py -v
```

138 tests covering engine functions, REST API endpoints, auth, rate limiting, body size limits, input guardrails, request tracing, error payloads, config toggle wiring, and version visibility.

### Linting and formatting

```bash
ruff check .         # lint
ruff format --check . # format verification
```

Configuration lives in `pyproject.toml`. CI runs both checks on every push and PR.

## Logging & Observability

### Structured logs

All request and error events are emitted as structured log lines via Python `logging`. Control verbosity with `GB_LOG_LEVEL` (default `INFO`).

Example request log:

```text
{"time":"...","level":"INFO","logger":"geometric_brain","message":"request method=POST path=/v1/brain/health-check status=200 latency_ms=12.3 request_id=abc123 version=1.0.0"}
```

Warning-level events: `auth_failed`, `rate_limited`, `payload_too_large`, `validation_error`, `http_error`.  
Error-level events: `unhandled_error` (includes `error_class`).

### Startup banner

On boot the API logs deployment-relevant config:

```text
startup service=geometric-brain version=1.0.0 env=production host=0.0.0.0 port=10000 auth=True rate_limit=True log_level=WARNING
```

### Request ID

Every response includes an `X-Request-ID` header. If the client sends `X-Request-ID`, it is echoed back; otherwise a UUID is generated. The same ID appears in:

- the response header (`X-Request-ID`)
- the response body (`request_id` field)
- all log lines for that request

### Standardized error payloads

All error responses follow a consistent shape:

```json
{
  "status": "error",
  "error_code": "VALIDATION_ERROR",
  "detail": "...",
  "request_id": "abc123",
  "schema_version": "1.0.0"
}
```

| Status | `error_code` | Trigger |
| --- | --- | --- |
| 401 | `UNAUTHORIZED` | Missing / invalid API key |
| 413 | `PAYLOAD_TOO_LARGE` | Body exceeds `MAX_BODY_BYTES` |
| 422 | `VALIDATION_ERROR` | Pydantic validation failure |
| 422 | `HTTP_ERROR` | Engine `ValueError` |
| 429 | `RATE_LIMITED` | Exceeds `RATE_LIMIT_RPM` |
| 500 | `INTERNAL_ERROR` | Unhandled exception |

## Deployment (Render)

This repo includes a Render blueprint (`render.yaml`) for one-click deployment.

### Quick deploy

1. Push this repo to GitHub.
2. Go to [Render Dashboard → New → Blueprint](https://dashboard.render.com/select-repo?type=blueprint).
3. Connect the repo. Render reads `render.yaml` and creates the service.
4. Set production env vars in the Render dashboard:

```bash
GB_AUTH_ENABLED=true
GB_API_KEYS=<your-secret-key>
GB_CORS_ORIGINS=https://your-app.example.com
GB_RATE_LIMIT_ENABLED=true
GB_RATE_LIMIT_RPM=30
GB_LOG_LEVEL=WARNING
GB_ENVIRONMENT=production
```

1. Deploy. Render runs `pip install -r requirements-ci.txt` and starts `python api.py`.

### Health probes

| Endpoint | Purpose |
| ---------- | --------- |
| `GET /healthz` | Liveness — returns `{"status":"ok"}` |
| `GET /readyz` | Readiness — returns `{"status":"ready","schema_version":"..."}` |
| `GET /v1/meta/capabilities` | API metadata (requires auth if enabled) |

### Smoke test

```bash
RENDER_URL="https://geometric-brain.onrender.com"
curl -sf "$RENDER_URL/healthz"
curl -sf "$RENDER_URL/readyz"
```

See [MVP_LAUNCH_CHECKLIST.md](MVP_LAUNCH_CHECKLIST.md) for the full deployment runbook, smoke tests, rollback procedures, and monitoring checklist.

## MCP Security Notes

The MCP server (`server.py`) runs over **stdio** by default, which inherits the trust boundary of the host process — typically Claude Desktop or another local MCP client. In this mode, access control is the responsibility of the host.

When running in **HTTP mode** (`--http`), the MCP server is network-accessible and does **not** enforce authentication. It should only be exposed on trusted networks or behind a reverse proxy that handles auth.

**Trust assumptions:**

- stdio mode: trusted local process, no network exposure
- HTTP mode: no built-in auth — treat as an internal service or add a proxy
- The MCP server does not share the REST API's auth middleware
- Do not expose the MCP HTTP endpoint to the public internet without additional protection

## Operational Reference

| Document | Contents |
| ---------- | ---------- |
| [PACKAGE_DELIVERY.md](PACKAGE_DELIVERY.md) | Runtime commands for setup, run modes, API calls, testing, release |
| [MVP_LAUNCH_CHECKLIST.md](MVP_LAUNCH_CHECKLIST.md) | Full launch runbook: pre-release, release, deploy, rollback, monitoring |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [RELEASE_NOTES.md](RELEASE_NOTES.md) | Current release highlights |
| [DEFERRED_WORK.md](DEFERRED_WORK.md) | Out-of-scope items recorded for future consideration |

## License

MIT License. See `LICENSE`.
