# Geometric Brain MCP

Production-ready spectral diagnostics for language model behavior, exposed through MCP tools, REST endpoints, and direct Python functions.

## Overview

Geometric Brain MCP evaluates model state quality by analyzing spectral structure in generated text or latent representations. It is designed for:

- runtime observability,
- intervention signal generation,
- side-by-side model comparison,
- integration into agent systems and platform services.

Core outputs include spectral regime estimates, confidence values, drift signals, and recommended correction direction.

## In Plain Terms

This program is a health monitor for AI model behavior.

1. It measures whether model output patterns look stable and coherent.
2. It flags when behavior appears to drift toward lower-quality dynamics.
3. It suggests a correction direction (increase/decrease pressure) from current spectral state.
4. It compares two models and tells you which one looks healthier under the same metric.

If you do not need the theory, you can treat it as: input model signals, get a reliability/quality diagnostic and action guidance.

## Interface Options

| Interface | Best for | Entry point |
| --- | --- | --- |
| MCP Server | MCP-compatible agents and orchestration runtimes | `python server.py` |
| HTTP API | Service-to-service and language-agnostic clients | `python api.py` |
| Python Module | Direct embedding in pipelines and research code | `import spectral_engine` |

## Feature Set

### `brain_health_check`

Text proxy analysis that estimates spectral health, regime, confidence, and warnings.

### `brain_manifold_audit`

Primary diagnostic for hidden states or eigenvalues, with strict validation and structured audit output.

### `brain_compute_correction`

Converts current `r_ratio` into a bounded intervention delta and recommendation.

### `brain_compare_models`

Runs parallel diagnostics and returns comparative health verdicts and deltas.

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start MCP server (stdio)

```bash
python server.py
```

### 3) Start MCP server (HTTP mode)

```bash
python server.py --http --host 0.0.0.0 --port 8000
```

### 4) Start REST API sidecar

```bash
python api.py
```

### 5) Use as a Python library

```python
from spectral_engine import spectral_health_check, manifold_audit

result = spectral_health_check("Example text for spectral analysis.")
print(result["status"], result["r_ratio"], result["regime"])
```

## Constants and Interpretation

- `GUE_R = 0.578`: target regime associated with strong level repulsion.
- `POISSON_R = 0.386`: baseline associated with weak repulsion.

Higher-level interpretation should be done in context of your model, prompts, and operating constraints.

## Validation and Testing

- End-to-end suite: `test_e2e.py`
- Current status target: all tests passing before release/tag.

## Deployment

- Render blueprint included: `render.yaml`
- Health probes: `/healthz`, `/readyz`
- API metadata: `/v1/meta/capabilities`

## Runtime Command Guide

Complete operational commands for install, run, test, API calls, and release workflows are documented in:

- `PACKAGE_DELIVERY.md`

## Docs Index

Quick links into the delivery guide:

- Setup and install: [PACKAGE_DELIVERY.md#1-environment-setup](PACKAGE_DELIVERY.md#1-environment-setup)
- Run modes (MCP/API): [PACKAGE_DELIVERY.md#2-run-modes](PACKAGE_DELIVERY.md#2-run-modes)
- Health and metadata checks: [PACKAGE_DELIVERY.md#3-health-and-metadata-endpoints](PACKAGE_DELIVERY.md#3-health-and-metadata-endpoints)
- Feature API commands: [PACKAGE_DELIVERY.md#4-feature-endpoints-rest](PACKAGE_DELIVERY.md#4-feature-endpoints-rest)
- Python usage: [PACKAGE_DELIVERY.md#5-python-usage](PACKAGE_DELIVERY.md#5-python-usage)
- Testing and validation: [PACKAGE_DELIVERY.md#6-testing-and-validation](PACKAGE_DELIVERY.md#6-testing-and-validation)
- Bridge validation: [PACKAGE_DELIVERY.md#7-bridge-validation-script](PACKAGE_DELIVERY.md#7-bridge-validation-script)
- Deployment checks: [PACKAGE_DELIVERY.md#8-containerplatform-delivery-render](PACKAGE_DELIVERY.md#8-containerplatform-delivery-render)
- Release commands: [PACKAGE_DELIVERY.md#9-release-commands](PACKAGE_DELIVERY.md#9-release-commands)
- Troubleshooting: [PACKAGE_DELIVERY.md#10-troubleshooting](PACKAGE_DELIVERY.md#10-troubleshooting)

## License

MIT License. See `LICENSE`.
