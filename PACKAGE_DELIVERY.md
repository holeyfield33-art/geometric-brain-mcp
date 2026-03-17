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
    "r_ratio": 0.51,
    "target_r": 0.578,
    "gain": 1.0,
    "clamp": 0.25
  }'
```

### 4.5 brain_compare_models

```bash
curl -s -X POST http://localhost:8001/v1/brain/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "left": {
      "source_type": "eigenvalues",
      "eigenvalues": [0.8, 1.0, 1.2, 1.5, 1.7]
    },
    "right": {
      "source_type": "eigenvalues",
      "eigenvalues": [0.9, 1.2, 1.4, 1.6, 1.9]
    },
    "left_label": "model_a",
    "right_label": "model_b"
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

corr = compute_correction(r_ratio=0.52)
print("correction:", corr["recommendation"], corr["delta"])

cmp_result = compare_models(
    left={"source_type": "eigenvalues", "eigenvalues": [1.0, 1.2, 1.5, 1.7]},
    right={"source_type": "eigenvalues", "eigenvalues": [0.9, 1.1, 1.3, 1.6]},
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

Use this when GPU/LLM dependencies are installed and available.

```bash
python bridge_validation.py
```

Expected artifact:

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

### Check status

```bash
git status
```

### Commit release changes

```bash
git add README.md PACKAGE_DELIVERY.md
git commit -m "docs: professional README and runtime delivery guide"
```

### Tag release

```bash
git tag -a v1.0.1 -m "v1.0.1 docs refresh"
```

### Push main and tags

```bash
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
