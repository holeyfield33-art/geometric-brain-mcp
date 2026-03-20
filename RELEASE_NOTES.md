# Release Notes — v1.1.1

**Date:** 2026-03-20
**Type:** Patch release (deployment fix, no API changes)

## Summary

Fixes a Render deployment crash caused by Python 3.14 incompatibility with MCP/Pydantic. Pins Python 3.12.8, updates the start command to use uvicorn directly, and tightens the Python version constraint.

## Changes

### Python version pinned to 3.12.8

Render now defaults to Python 3.14, which breaks MCP and Pydantic at import time. This release pins 3.12.8 via:
- `.python-version` file in the repo root
- `PYTHON_VERSION` env var in `render.yaml`
- `requires-python = ">=3.11,<3.14"` in `pyproject.toml`

### Start command updated

The Render start command changed from `python api.py` to:

```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

This gives Render proper process control and ensures the app binds to the correct host and port.

### render.yaml cleanup

Removed the deprecated `pythonVersion: "3.11.12"` field, replaced with the `PYTHON_VERSION` env var approach (more reliable on Render).

## Upgrade path

This is a **non-breaking** patch release. No API, MCP, or library changes. Existing integrations are unaffected.

## Version

| Source | Value |
| -------- | ------- |
| `config.SCHEMA_VERSION` | `1.1.1` |
| `pyproject.toml [project] version` | `1.1.1` |
| Git tag | `v1.1.1` |
