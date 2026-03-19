# MVP Launch Checklist

Human-executable runbook for releasing Geometric Brain MCP v1.1.0.
Every command is copy-pasteable. No guessing required.

---

## Pre-release

### 1. Clean working tree

```bash
cd geometric-brain-mcp
git status
# Must show no unexpected changes. All Phase 0–9 work should be staged.
```

### 2. Install dependencies

```bash
pip install -r requirements-ci.txt
```

### 3. Lint and format check

```bash
ruff check .
ruff format --check .
# Both must exit 0 with no errors.
```

### 4. Compile check

```bash
python -m py_compile spectral_engine.py api.py server.py config.py
# Silent output = success.
```

### 5. Run full test suite

```bash
python -m pytest test_e2e.py -v
# Expected: 138 passed, 0 failed, 0 warnings.
```

### 6. Verify version strings match

```bash
python -c "import config; print('config:', config.SCHEMA_VERSION)"
grep '^version' pyproject.toml
# Both must show "1.1.0".
```

### 7. Verify auth works

```bash
python -m pytest test_e2e.py -v -k "TestAuthMiddleware or TestAuthDisabled" --tb=short
# Expected: 13 passed.
```

### 8. Verify rate limiting works

```bash
python -m pytest test_e2e.py -v -k "TestRateLimiting or TestRateLimitDisabled" --tb=short
# Expected: 5 passed.
```

### 9. Verify input guardrails work

```bash
python -m pytest test_e2e.py -v -k "TestInputGuardrails or TestBodySizeLimit or TestConfigToggles" --tb=short
# Expected: 14 passed.
```

---

## Release

### 10. Version bump

Update version in **both** files to `1.1.0`:

| File | Field | Command |
| ------ | ------- | --------- |
| `config.py` | `SCHEMA_VERSION` | `sed -i 's/SCHEMA_VERSION: str = "1.0.0"/SCHEMA_VERSION: str = "1.1.0"/' config.py` |
| `pyproject.toml` | `version` | `sed -i 's/version = "1.0.0"/version = "1.1.0"/' pyproject.toml` |

Verify:

```bash
grep SCHEMA_VERSION config.py
grep '^version' pyproject.toml
```

### 11. Commit and tag

```bash
git add -A
git commit -m "release: v1.1.0 — config, auth, rate limiting, logging, CI, 138 tests"
git tag -a v1.1.0 -m "v1.1.0 closed MVP"
```

### 12. Push

```bash
git push origin main
git push origin v1.1.0
```

### 13. Verify CI passes

Go to: `https://github.com/holeyfield33-art/geometric-brain-mcp/actions`

Wait for the CI workflow to complete on the `v1.1.0` push. Both Python 3.11 and 3.12 jobs must be green.

### 14. Create GitHub release

```bash
gh release create v1.1.0 \
  --title "v1.1.0 — Closed MVP" \
  --notes-file RELEASE_NOTES.md
```

Or create manually at: `https://github.com/holeyfield33-art/geometric-brain-mcp/releases/new`

- Tag: `v1.1.0`
- Title: `v1.1.0 — Closed MVP`
- Body: paste contents of `RELEASE_NOTES.md`

### 15. Build package

```bash
pip install build twine
python -m build
```

Expected output:

```text
Successfully built geometric_brain_mcp-1.1.0.tar.gz and geometric_brain_mcp-1.1.0-py3-none-any.whl
```

Verify artifacts:

```bash
ls dist/
# geometric_brain_mcp-1.1.0.tar.gz
# geometric_brain_mcp-1.1.0-py3-none-any.whl
```

### 16. Validate package

```bash
python -m twine check dist/*
# Both files must show PASSED.
```

### 17. Publish to PyPI (test first)

**Test PyPI (dry run):**

```bash
python -m twine upload --repository testpypi dist/*
# Requires a TestPyPI account and API token.
# Set token: export TWINE_PASSWORD=pypi-AgEI...
# Or use ~/.pypirc with [testpypi] section.
```

Verify on TestPyPI: `https://test.pypi.org/project/geometric-brain-mcp/1.1.0/`

**Production PyPI:**

```bash
python -m twine upload dist/*
# Requires a PyPI account and API token.
# Set token: export TWINE_PASSWORD=pypi-AgEI...
# Or use ~/.pypirc with [pypi] section.
```

Verify on PyPI: `https://pypi.org/project/geometric-brain-mcp/1.1.0/`

### 18. Verify install from PyPI

```bash
pip install geometric-brain-mcp==1.1.0
python -c "import spectral_engine; import config; print(config.SCHEMA_VERSION)"
# Expected: 1.1.0
```

---

## Deployment (Render)

### 19. Set production environment variables

In the Render dashboard for the `geometric-brain` service, set:

```bash
GB_AUTH_ENABLED=true
GB_API_KEYS=<your-production-api-key>
GB_CORS_ORIGINS=https://your-app.example.com
GB_RATE_LIMIT_ENABLED=true
GB_RATE_LIMIT_RPM=30
GB_LOG_LEVEL=WARNING
GB_ENVIRONMENT=production
```

Leave `PORT` as configured by Render (typically 10000, set in `render.yaml`).

### 20. Deploy

Render auto-deploys on push to main (if configured), or trigger a manual deploy from the Render dashboard.

Startup command (from `render.yaml`):

```bash
python api.py
```

### 21. Deployment smoke tests

Wait for the service to start (check Render logs for the startup banner), then run **all** of these:

```bash
RENDER_URL="https://geometric-brain.onrender.com"

# Health probe
curl -sf "$RENDER_URL/healthz" && echo " OK"
# Expected: {"status":"ok","service":"geometric-brain"}

# Readiness probe
curl -sf "$RENDER_URL/readyz" && echo " OK"
# Expected: {"status":"ready","schema_version":"1.1.0"}

# Capabilities (requires auth if enabled)
curl -sf "$RENDER_URL/v1/meta/capabilities" \
  -H "Authorization: Bearer <your-production-api-key>" && echo " OK"
# Expected: JSON with tools, constants, limits
```

### 22. Functional smoke tests

```bash
# Health check (text proxy)
curl -sf -X POST "$RENDER_URL/v1/brain/health-check" \
  -H "Authorization: Bearer <your-production-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"text": "The model output appears consistent and well-formed."}' | python -m json.tool
# Expected: 200 with status, r_ratio, regime, schema_version

# Compute correction
curl -sf -X POST "$RENDER_URL/v1/brain/compute-correction" \
  -H "Authorization: Bearer <your-production-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"current_r_ratio": 0.52}' | python -m json.tool
# Expected: 200 with delta, direction, recommended_action

# Auth rejection (no key)
curl -s -X POST "$RENDER_URL/v1/brain/health-check" \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}' | python -m json.tool
# Expected: 401 with error_code UNAUTHORIZED

# Rate limit verification (burst 31+ requests)
for i in $(seq 1 35); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$RENDER_URL/v1/brain/compute-correction" \
    -H "Authorization: Bearer <your-production-api-key>" \
    -H "Content-Type: application/json" \
    -d '{"current_r_ratio": 0.5}')
  if [ "$CODE" = "429" ]; then echo "Rate limit triggered at request $i: OK"; break; fi
done
# Expected: 429 at or before request 31 (if GB_RATE_LIMIT_RPM=30)

# Manifold audit (eigenvalue mode)
curl -sf -X POST "$RENDER_URL/v1/brain/manifold-audit" \
  -H "Authorization: Bearer <your-production-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "eigenvalues", "eigenvalues": [0.9, 1.1, 1.4, 1.8, 2.1]}' | python -m json.tool
# Expected: 200 with spectral_health_score, mean_r_ratio, spectral_regime

# Compare models
curl -sf -X POST "$RENDER_URL/v1/brain/compare-models" \
  -H "Authorization: Bearer <your-production-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "left": {"model_label": "a", "source_type": "eigenvalues", "eigenvalues": [0.8, 1.0, 1.2, 1.5]},
    "right": {"model_label": "b", "source_type": "eigenvalues", "eigenvalues": [0.9, 1.2, 1.4, 1.6]}
  }' | python -m json.tool
# Expected: 200 with healthier_model, delta_health_score
```

**Pass criteria:** all 4 POST endpoints return 200, auth rejection returns 401, rate limit triggers at expected threshold. schema_version in all responses must be "1.1.0".

---

## Post-launch

### 23. Rollback

If something goes wrong after deploy:

**Option A — Revert to previous tag (v1.0.2):**

```bash
git checkout v1.0.2
git push origin HEAD:main --force-with-lease
```

Then trigger a Render redeploy.

**Option B — Disable new features without rollback:**

Set these environment variable overrides in Render:

```bash
GB_AUTH_ENABLED=false
GB_RATE_LIMIT_ENABLED=false
GB_LOG_LEVEL=INFO
```

This returns the service to pre-hardening behavior without reverting code.

**Option C — Revert the tag only:**

```bash
git tag -d v1.1.0
git push origin :refs/tags/v1.1.0
```

### 24. Logs to inspect

Render logs are available in the dashboard. Key log events to watch:

| Event | Level | Meaning |
| ------- | ------- | --------- |
| `startup service=geometric-brain ...` | INFO | Service booted with correct config |
| `request method=POST path=/v1/brain/... status=200` | INFO | Normal traffic |
| `auth_failed` | WARNING | Someone sent a bad/missing API key |
| `rate_limited` | WARNING | Client exceeded RPM limit |
| `payload_too_large` | WARNING | Oversized request rejected |
| `validation_error` | WARNING | Malformed request body |
| `unhandled_error` | ERROR | Bug — investigate immediately |

### 25. Common failure points

| Symptom | Likely cause | Fix |
| --------- | ------------- | ----- |
| 401 on all requests | `GB_AUTH_ENABLED=true` but `GB_API_KEYS` not set or wrong | Set correct keys in Render env vars |
| 429 immediately | `GB_RATE_LIMIT_RPM` set too low | Increase RPM or disable rate limiting |
| 500 on startup | Missing dependency | Check Render build logs; ensure `requirements-ci.txt` installs cleanly |
| Import error for `config` | Render using old code before `config.py` existed | Trigger a fresh deploy from latest main |
| Port conflict | `PORT` env var not set | Render sets this automatically; check `render.yaml` |

### 26. First 24 hours monitoring

**Hour 0–1 (immediately after deploy):**

- [ ] Startup banner appeared in logs with correct version and config
- [ ] `GET /healthz` returns `{"status": "ok"}`
- [ ] `GET /readyz` returns `schema_version: "1.1.0"`
- [ ] All 4 POST endpoints return 200 with valid auth
- [ ] Auth rejection (no key) returns 401
- [ ] Rate limit triggers at configured threshold

**Hour 1–6:**

- [ ] No `unhandled_error` (level=ERROR) events in logs
- [ ] `auth_failed` events are from bots/scanners, not your own clients
- [ ] Response latency < 100ms p95 (check Render metrics)
- [ ] No 500 status codes in request logs

**Hour 6–24:**

- [ ] Rate limit store is not growing unbounded (in-memory, resets per window)
- [ ] No repeated `validation_error` from the same client (indicates integration bug)
- [ ] At least one successful call to each POST endpoint from intended client
- [ ] Render free tier spin-down/spin-up cycle works (cold start < 30s)

**Red flags requiring immediate action:**

- Any `unhandled_error` → check stack trace in logs, hotfix or rollback
- All requests returning 401 → `GB_API_KEYS` misconfigured in Render env vars
- All requests returning 429 → `GB_RATE_LIMIT_RPM` too low or store not clearing
- Import errors in logs → missing `config.py` in deploy (stale build cache)

---

## Checklist summary

| # | Step | Command / action | Expected |
| --- | ------ | ----------------- | ---------- |
| 1 | Clean tree | `git status` | No unexpected changes |
| 2 | Install | `pip install -r requirements-ci.txt` | Success |
| 3 | Lint | `ruff check . && ruff format --check .` | All passed |
| 4 | Compile | `python -m py_compile ...` | Silent |
| 5 | Test | `python -m pytest test_e2e.py -v` | 138 passed |
| 6 | Version | Check config.py + pyproject.toml | Both 1.1.0 |
| 7–9 | Auth/limits | Targeted pytest runs | All green |
| 10 | Bump | sed commands | Both files updated |
| 11 | Commit + tag | `git commit` + `git tag` | v1.1.0 |
| 12 | Push | `git push` (main + tag) | Remote updated |
| 13 | CI | GitHub Actions | Green on 3.11 + 3.12 |
| 14 | GitHub release | `gh release create` | Published |
| 15 | Build package | `python -m build` | .tar.gz + .whl |
| 16 | Validate package | `twine check dist/*` | PASSED |
| 17 | Publish to PyPI | `twine upload dist/*` | Live on PyPI |
| 18 | Verify install | `pip install geometric-brain-mcp==1.1.0` | Imports OK |
| 19 | Env vars | Render dashboard | Production config set |
| 20 | Deploy | Render | Service running |
| 21–22 | Smoke tests | curl commands | All endpoints respond correctly |
| 23 | Rollback plan | Options A/B/C documented | Ready |
| 24–25 | Log/failure review | Render logs | No ERROR events |
| 26 | 24h monitoring | Checklist above | All items checked |
