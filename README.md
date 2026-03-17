# Geometric Brain MCP

**Standalone spectral health node for AI agents.**

Plug the Geometric Brain into any agent via MCP, REST API, or Python import.
It monitors transformer latent space health using GUE spectral rigidity
and returns actionable intervention signals.

## Three Doors, One Brain

| Door | Who it's for | How to connect |
|------|-------------|----------------|
| **MCP Server** | Claude Desktop, MCP-compatible agents | `python server.py` (stdio) or `python server.py --http` |
| **Sidecar API** | Any device, any language, any framework | `POST /v1/brain/health-check` - deploy to Render |
| **PyPI Import** | ML engineers embedding in training loops | `from spectral_engine import manifold_audit` |

## Tools

### `brain_health_check`
Text in, spectral health score out. Lightweight proxy measurement.

### `brain_manifold_audit`
Hidden state vectors or eigenvalues in, full manifold analysis out.
The core diagnostic.

### `brain_compute_correction`
Current r-ratio in, intervention signal out. The feedback loop.

### `brain_compare_models`
Two models in, comparative verdict out. The decision tool. (Premium)

## Quick Start

### MCP (Local)
```bash
pip install -r requirements.txt
python server.py
```

### MCP (Remote)
```bash
python server.py --http --port 8000
```

### Sidecar API (Render)
```bash
# Deploy with render.yaml or:
pip install -r requirements.txt
python api.py
```

### Python Import
```python
from spectral_engine import spectral_health_check, manifold_audit

result = spectral_health_check("your text here")
print(result["r_ratio"], result["regime"])
```

## The Math

SHI = <r> / (F x RTI)

The spacing ratio <r> measures eigenvalue repulsion in the latent manifold:
- **<r> = 0.578** - GUE rigidity (coherent reasoning)
- **<r> = 0.386** - Poisson spacing (context decoherence / hallucination)

The k=1 invariant from the Berry-Keating Hamiltonian determines the
dynamic sigma in the heat kernel, maintaining Cheeger constant > 0.

Full theory: [geometric-brain](https://github.com/holeyfield33-art/geometric-brain)
Implementation: [unitarity-lab](https://github.com/holeyfield33-art/unitarity-lab)

## Built Through TMRP

- **Gemini** - core spectral functions (extracted from unitarity-lab)
- **ChatGPT** - Pydantic schemas, edge case analysis, tier architecture
- **Claude** - orchestration, merge, MCP/API server scaffold

(c) 2026 Aletheia Sovereign Systems - MIT License
