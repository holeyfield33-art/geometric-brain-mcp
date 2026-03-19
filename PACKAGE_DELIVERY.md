# Package Delivery and Runtime Commands

This document provides the full runtime command set for setup, execution, validation, API usage, and release actions.

## 1. Environment Setup

### Clone and enter project

```bash
git clone https://github.com/holeyfield33-art/geometric-brain-mcp.git
cd geometric-brain-mcp
```

### Optional virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Modes

### MCP server (stdio mode)

```bash
python server.py
```

### MCP server (HTTP mode)

```bash
python server.py --http --host 0.0.0.0 --port 8000
```

### REST API sidecar

```bash
python api.py
```

### REST API sidecar with uvicorn

```bash
uvicorn api:app --host 0.0.0.0 --port 8001
```

## 3. Health and Metadata Endpoints

### Liveness

```bash
curl -s http://localhost:8001/healthz
```

### Readiness

```bash
curl -s http://localhost:8001/readyz
```

### Capabilities

```bash
curl -s http://localhost:8001/v1/meta/capabilities
```

## 4. Feature Endpoints (REST)

### 4.1 brain_health_check

```bash
curl -s -X POST http://localhost:8001/v1/brain/health-check \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The model response appears stable and consistent.",
    "window_size": 64,
    "stride": 16
  }'
```

### 4.2 brain_manifold_audit (hidden states)

```bash
curl -s -X POST http://localhost:8001/v1/brain/manifold-audit \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "hidden_states",
    "hidden_states": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.3, 0.2]],
    "center": true,
    "normalize": true,
    "return_eigenvalues": false
  }'
```

### 4.3 brain_manifold_audit (eigenvalues)

```bash
curl -s -X POST http://localhost:8001/v1/brain/manifold-audit \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "eigenvalues",
    "eigenvalues": [0.9, 1.1, 1.3, 1.6, 2.0]
  }'
```

### 4.4 brain_compute_correction

```bash
curl -s -X POST http://localhost:8001/v1/brain/compute-correction \
  -H "Content-Type: application/json" \
  -d '{
    "current_r_ratio": 0.51,
    "target_r_ratio": 0.578,
    "gain": 1.0,
    "clamp_output": true,
    "max_magnitude": 1.0
  }'
```

### 4.5 brain_compare_models

```bash
curl -s -X POST http://localhost:8001/v1/brain/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "left": {
      "model_label": "model_a",
      "source_type": "eigenvalues",
      "eigenvalues": [0.8, 1.0, 1.2, 1.5, 1.7]
    },
    "right": {
      "model_label": "model_b",
      "source_type": "eigenvalues",
      "eigenvalues": [0.9, 1.2, 1.4, 1.6, 1.9]
    }
  }'
```

## 5. Python Usage

### Basic import and execution

```bash
python - <<'PY'
from spectral_engine import spectral_health_check, manifold_audit, compute_correction, compare_models

text_result = spectral_health_check("Example text for analysis")
print("health_check:", text_result["status"], text_result.get("r_ratio"))

aud_result = manifold_audit(eigenvalues=[0.9, 1.1, 1.4, 1.8, 2.1])
print("audit:", aud_result["status"], aud_result.get("spectral_health_score"))

corr = compute_correction(current_r_ratio=0.52)
print("correction:", corr["recommended_action"], corr["delta"])

cmp_result = compare_models(
    left_eigenvalues=[1.0, 1.2, 1.5, 1.7],
    right_eigenvalues=[0.9, 1.1, 1.3, 1.6],
)
print("compare:", cmp_result["healthier_model"], cmp_result["delta_health_score"])
PY
```

## 6. Testing and Validation

### Run full e2e tests

```bash
python -m pytest test_e2e.py -v
```

### Run all tests in repository

```bash
python -m pytest -v
```

### Compile/syntax check

```bash
python -m py_compile spectral_engine.py server.py api.py bridge_validation.py
```

## 7. Bridge Validation Script

**Research tool, not a runtime component.** This script tests whether eigenvalue spacing on real model hidden states can separate truthful from hallucinated text. It requires GPU-capable hardware and heavy dependencies (`torch`, `transformers`, `datasets`, `scikit-learn`) that are **not** included in `requirements-ci.txt`.

No committed results exist in this repository. If you run the script and commit the output, that constitutes evidence for your specific model and dataset — not a general claim.

### Prerequisites

```bash
pip install torch transformers datasets scikit-learn accelerate
```

### Run

```bash
python bridge_validation.py
```

### Expected artifact

```bash
ls -lh hidden_state_validation.json
```

## 8. Container/Platform Delivery (Render)

### Local parity check using render command style

```bash
PORT=10000 python api.py
```

### Render blueprint

```bash
cat render.yaml
```

## 9. Release Commands

### Pre-release checklist

Before every tagged release, run through these steps **in order**:

```bash
# 1. Ensure working tree is clean
git status

# 2. Run linter and formatter checks
ruff check .
ruff format --check .

# 3. Compile check all source files
python -m py_compile spectral_engine.py server.py api.py config.py

# 4. Run full test suite (must be 0 failures, 0 warnings)
python -m pytest test_e2e.py -v

# 5. Verify version strings are in sync
python -c "import config; print('config:', config.SCHEMA_VERSION)"
grep '^version' pyproject.toml
# Both must show the same value.
```

### Version bump

Update the version in **both** places (they must match):

| File | Field |
|------|-------|
| `config.py` | `SCHEMA_VERSION` |
| `pyproject.toml` | `[project] version` |

### Tag and push

```bash
git add -A
git commit -m "release: vX.Y.Z"
git tag -a vX.Y.Z -m "vX.Y.Z <summary>"
git push origin main
git push origin --tags
```

## 10. Troubleshooting

### Port already in use

```bash
lsof -i :8001
kill -9 <PID>
```

### Dependency mismatch

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Quick smoke test

```bash
curl -s http://localhost:8001/healthz && echo
python -m pytest test_e2e.py -q
```
