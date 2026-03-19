"""
Geometric Brain - Configuration
Centralized, environment-driven runtime config.

All settings read from environment variables with sane defaults.
Local dev works with zero configuration. Production behavior is
controlled entirely through env vars.

(c) 2026 Aletheia Sovereign Systems - MIT License
"""

import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


def _list(val: str) -> list[str]:
    """Parse comma-separated string into list, stripping whitespace."""
    return [s.strip() for s in val.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

HOST: str = os.environ.get("GB_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("GB_PORT", os.environ.get("PORT", "8000")))

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

AUTH_ENABLED: bool = _bool(os.environ.get("GB_AUTH_ENABLED", "false"))

# Comma-separated list of accepted API keys.
# Only checked when AUTH_ENABLED is true.
API_KEYS: list[str] = _list(os.environ.get("GB_API_KEYS", ""))

# Endpoints that remain public even when auth is enabled.
PUBLIC_PATHS: set[str] = {"/healthz", "/readyz", "/docs", "/openapi.json"}

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# Comma-separated origins. Default "*" for local dev; restrict in production.
CORS_ORIGINS: list[str] = _list(os.environ.get("GB_CORS_ORIGINS", "*"))

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT_ENABLED: bool = _bool(os.environ.get("GB_RATE_LIMIT_ENABLED", "false"))

# Max requests per window per client IP.
RATE_LIMIT_RPM: int = int(os.environ.get("GB_RATE_LIMIT_RPM", "60"))

# ---------------------------------------------------------------------------
# Request size caps
# ---------------------------------------------------------------------------

MAX_TEXT_LENGTH: int = int(os.environ.get("GB_MAX_TEXT_LENGTH", "20000"))
MAX_HIDDEN_STATE_SAMPLES: int = int(os.environ.get("GB_MAX_HIDDEN_STATE_SAMPLES", "8192"))
MAX_HIDDEN_STATE_DIMS: int = int(os.environ.get("GB_MAX_HIDDEN_STATE_DIMS", "4096"))
MAX_EIGENVALUES: int = int(os.environ.get("GB_MAX_EIGENVALUES", "65536"))

# Max request body in bytes.  Defence-in-depth before JSON parsing.
MAX_BODY_BYTES: int = int(os.environ.get("GB_MAX_BODY_BYTES", str(10 * 1024 * 1024)))  # 10 MB

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.environ.get("GB_LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Version / environment labels
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.1.0"
__version__: str = SCHEMA_VERSION  # convenience alias

ENVIRONMENT: str = os.environ.get("GB_ENVIRONMENT", "development")
