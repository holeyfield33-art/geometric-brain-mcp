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

import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectral_engine import (
    compare_models,
    compute_correction,
    manifold_audit,
    spectral_health_check,
)

SCHEMA_VERSION = "1.0.0"

app = FastAPI(
    title="Geometric Brain API",
    version=SCHEMA_VERSION,
    description="Spectral rigidity analysis for transformer latent spaces. GUE health monitoring for any AI agent.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schemas

class HealthCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=20_000)
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
        if self.hidden_states and len(self.hidden_states) > 8192:
            raise ValueError(f"Max 8192 samples, got {len(self.hidden_states)}")
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
            "max_text_length": 20_000,
            "max_hidden_state_samples": 8192,
            "max_hidden_state_dimensions": 4096,
            "max_eigenvalues": 65536,
        },
    }


@app.post("/v1/brain/health-check")
async def health_check(req: HealthCheckRequest):
    try:
        result = spectral_health_check(
            text=req.text,
            window_size=req.window_size,
            stride=req.stride,
        )
        result["schema_version"] = SCHEMA_VERSION
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/v1/brain/manifold-audit")
async def manifold_audit_endpoint(req: ManifoldAuditRequest):
    try:
        result = manifold_audit(
            hidden_states=req.hidden_states,
            eigenvalues=req.eigenvalues,
            normalize=req.normalize,
            center=req.center,
        )
        result["schema_version"] = SCHEMA_VERSION
        if not req.return_eigenvalues:
            result.pop("eigenvalues", None)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/v1/brain/compute-correction")
async def compute_correction_endpoint(req: CorrectionRequest):
    try:
        result = compute_correction(
            current_r_ratio=req.current_r_ratio,
            target_r_ratio=req.target_r_ratio,
            gain=req.gain,
            clamp=req.clamp_output,
            max_magnitude=req.max_magnitude,
        )
        result["schema_version"] = SCHEMA_VERSION
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/v1/brain/compare-models")
async def compare_models_endpoint(req: CompareRequest):
    try:
        result = compare_models(
            left_hidden_states=req.left.hidden_states,
            left_eigenvalues=req.left.eigenvalues,
            right_hidden_states=req.right.hidden_states,
            right_eigenvalues=req.right.eigenvalues,
            normalize=req.normalize,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["left_label"] = req.left.model_label
        result["right_label"] = req.right.model_label
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
