"""
Geometric Brain - Sidecar API
REST endpoint version. Deploy to Render, Railway, or any host.
Any device with internet can hit this - phone, tablet, laptop.

Endpoints map 1:1 to MCP tools:
  POST /v1/brain/health-check
  POST /v1/brain/manifold-audit
  POST /v1/brain/compute-correction
  POST /v1/brain/compare-models

(c) 2026 Aletheia Sovereign Systems - MIT License
"""

import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

import config
from spectral_engine import (
    compare_models,
    compute_correction,
    manifold_audit,
    spectral_health_check,
)

# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}')
)

logger = logging.getLogger("geometric_brain")
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
logger.addHandler(_log_handler)
logger.propagate = False

SCHEMA_VERSION = config.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown events
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI):
    logger.info(
        "startup service=geometric-brain version=%s env=%s host=%s port=%d auth=%s rate_limit=%s log_level=%s",
        SCHEMA_VERSION,
        config.ENVIRONMENT,
        config.HOST,
        config.PORT,
        config.AUTH_ENABLED,
        config.RATE_LIMIT_ENABLED,
        config.LOG_LEVEL,
    )
    yield


app = FastAPI(
    title="Geometric Brain API",
    version=SCHEMA_VERSION,
    description="Spectral rigidity analysis for transformer latent spaces. GUE health monitoring for any AI agent.",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = round((time.monotonic() - start) * 1000, 1)
    request_id = getattr(request.state, "request_id", "-")
    logger.info(
        "request method=%s path=%s status=%d latency_ms=%.1f request_id=%s version=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        request_id,
        SCHEMA_VERSION,
    )
    return response


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not config.AUTH_ENABLED:
        return await call_next(request)

    if request.url.path in config.PUBLIC_PATHS:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    else:
        token = ""

    if not token or token not in config.API_KEYS:
        request_id = getattr(request.state, "request_id", "-")
        logger.warning("auth_failed path=%s request_id=%s", request.url.path, request_id)
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "error_code": "UNAUTHORIZED",
                "detail": "Invalid or missing API key. Provide a valid key via the Authorization: Bearer <key> header.",
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
            },
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiting middleware
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not config.RATE_LIMIT_ENABLED:
        return await call_next(request)

    if request.url.path in config.PUBLIC_PATHS:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = 60.0

    timestamps = _rate_limit_store[client_ip]
    timestamps[:] = [t for t in timestamps if t > now - window]

    if len(timestamps) >= config.RATE_LIMIT_RPM:
        request_id = getattr(request.state, "request_id", "-")
        logger.warning(
            "rate_limited client=%s path=%s request_id=%s",
            client_ip,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "error_code": "RATE_LIMITED",
                "detail": f"Rate limit exceeded. Max {config.RATE_LIMIT_RPM} requests per minute.",
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
            },
        )

    timestamps.append(now)
    return await call_next(request)


# ---------------------------------------------------------------------------
# Body size middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def body_size_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > config.MAX_BODY_BYTES:
        request_id = getattr(request.state, "request_id", "-")
        logger.warning(
            "payload_too_large path=%s bytes=%s request_id=%s",
            request.url.path,
            content_length,
            request_id,
        )
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "error_code": "PAYLOAD_TOO_LARGE",
                "detail": f"Request body exceeds maximum size of {config.MAX_BODY_BYTES} bytes.",
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
            },
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Exception handlers — standardized error payloads
# ---------------------------------------------------------------------------


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    rid = _request_id(request)
    # Sanitize errors: convert non-serializable ctx values to strings.
    errors = []
    for err in exc.errors():
        clean = {k: v for k, v in err.items() if k != "ctx"}
        if "ctx" in err:
            clean["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
        errors.append(clean)
    logger.warning(
        "validation_error path=%s request_id=%s error=%s",
        request.url.path,
        rid,
        str(errors),
    )
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error_code": "VALIDATION_ERROR",
            "detail": errors,
            "request_id": rid,
            "schema_version": SCHEMA_VERSION,
        },
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    rid = _request_id(request)
    logger.warning(
        "http_error path=%s status=%d request_id=%s detail=%s",
        request.url.path,
        exc.status_code,
        rid,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": "HTTP_ERROR",
            "detail": exc.detail,
            "request_id": rid,
            "schema_version": SCHEMA_VERSION,
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    rid = _request_id(request)
    logger.error(
        "unhandled_error path=%s request_id=%s error_class=%s error=%s",
        request.url.path,
        rid,
        type(exc).__name__,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred.",
            "request_id": rid,
            "schema_version": SCHEMA_VERSION,
        },
    )


# Schemas


class HealthCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=config.MAX_TEXT_LENGTH)
    window_size: int = Field(default=128, ge=16, le=4096)
    stride: int = Field(default=64, ge=1, le=4096)


class ManifoldAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden_states: Optional[List[List[float]]] = None
    eigenvalues: Optional[List[float]] = None
    source_type: Literal["hidden_states", "eigenvalues"]
    normalize: bool = True
    center: bool = True
    return_eigenvalues: bool = False

    @model_validator(mode="after")
    def check_source(self):
        if self.source_type == "hidden_states" and not self.hidden_states:
            raise ValueError("hidden_states required when source_type='hidden_states'")
        if self.source_type == "eigenvalues" and not self.eigenvalues:
            raise ValueError("eigenvalues required when source_type='eigenvalues'")
        if self.hidden_states:
            if len(self.hidden_states) > config.MAX_HIDDEN_STATE_SAMPLES:
                raise ValueError(f"Max {config.MAX_HIDDEN_STATE_SAMPLES} samples, got {len(self.hidden_states)}")
            if self.hidden_states[0] and len(self.hidden_states[0]) > config.MAX_HIDDEN_STATE_DIMS:
                raise ValueError(f"Max {config.MAX_HIDDEN_STATE_DIMS} dimensions, got {len(self.hidden_states[0])}")
        if self.eigenvalues and len(self.eigenvalues) > config.MAX_EIGENVALUES:
            raise ValueError(f"Max {config.MAX_EIGENVALUES} eigenvalues, got {len(self.eigenvalues)}")
        return self


class CorrectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_r_ratio: float = Field(..., ge=0.0, le=1.0)
    target_r_ratio: float = Field(default=0.578, ge=0.0, le=1.0)
    gain: float = Field(default=1.0, ge=0.0, le=10.0)
    clamp_output: bool = True
    max_magnitude: float = Field(default=1.0, gt=0.0, le=100.0)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_label: str = Field(..., min_length=1, max_length=128)
    hidden_states: Optional[List[List[float]]] = None
    eigenvalues: Optional[List[float]] = None
    source_type: Literal["hidden_states", "eigenvalues"] = "hidden_states"


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: ModelSpec
    right: ModelSpec
    normalize: bool = True


# Endpoints


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "geometric-brain"}


@app.get("/readyz")
async def readyz():
    return {"status": "ready", "schema_version": SCHEMA_VERSION}


@app.get("/v1/meta/capabilities")
async def capabilities():
    return {
        "schema_version": SCHEMA_VERSION,
        "tools": [
            "brain_health_check",
            "brain_manifold_audit",
            "brain_compute_correction",
            "brain_compare_models",
        ],
        "constants": {
            "gue_r": 0.578,
            "poisson_r": 0.386,
        },
        "limits": {
            "max_text_length": config.MAX_TEXT_LENGTH,
            "max_hidden_state_samples": config.MAX_HIDDEN_STATE_SAMPLES,
            "max_hidden_state_dimensions": config.MAX_HIDDEN_STATE_DIMS,
            "max_eigenvalues": config.MAX_EIGENVALUES,
            "max_body_bytes": config.MAX_BODY_BYTES,
        },
    }


@app.post("/v1/brain/health-check")
async def health_check(request: Request, req: HealthCheckRequest):
    try:
        result = spectral_health_check(
            text=req.text,
            window_size=req.window_size,
            stride=req.stride,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = _request_id(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.post("/v1/brain/manifold-audit")
async def manifold_audit_endpoint(request: Request, req: ManifoldAuditRequest):
    try:
        result = manifold_audit(
            hidden_states=req.hidden_states,
            eigenvalues=req.eigenvalues,
            normalize=req.normalize,
            center=req.center,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = _request_id(request)
        if not req.return_eigenvalues:
            result.pop("eigenvalues", None)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.post("/v1/brain/compute-correction")
async def compute_correction_endpoint(request: Request, req: CorrectionRequest):
    try:
        result = compute_correction(
            current_r_ratio=req.current_r_ratio,
            target_r_ratio=req.target_r_ratio,
            gain=req.gain,
            clamp=req.clamp_output,
            max_magnitude=req.max_magnitude,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = _request_id(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.post("/v1/brain/compare-models")
async def compare_models_endpoint(request: Request, req: CompareRequest):
    try:
        result = compare_models(
            left_hidden_states=req.left.hidden_states,
            left_eigenvalues=req.left.eigenvalues,
            right_hidden_states=req.right.hidden_states,
            right_eigenvalues=req.right.eigenvalues,
            normalize=req.normalize,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = _request_id(request)
        result["left_label"] = req.left.model_label
        result["right_label"] = req.right.model_label
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
