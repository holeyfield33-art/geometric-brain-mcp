"""
Geometric Brain MCP Server
Standalone spectral health node for any AI agent.

Plug this into Claude Desktop, any MCP-compatible client,
or run as a remote server via streamable HTTP.

Built through TMRP: Gemini (core math) + ChatGPT (schemas/architecture) + Claude (merge/server)
(c) 2026 Aletheia Sovereign Systems - MIT License
"""

import json
from typing import Any, List, Literal, Optional, cast

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectral_engine import (
    compare_models,
    compute_correction,
    manifold_audit,
    spectral_health_check,
)

# Server
mcp = FastMCP("geometric_brain_mcp")

SCHEMA_VERSION = "1.0.0"


def _tool_annotations(title: str) -> Any:
    return cast(
        Any,
        {
            "title": title,
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )


# ==============================================
# Input / Output Schemas (from ChatGPT relay)
# ==============================================


# Tool 1: Health Check

class HealthCheckInput(BaseModel):
    """Analyze raw text for spectral health indicators."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="Raw input text to analyze",
    )
    window_size: int = Field(
        default=128,
        ge=16,
        le=4096,
        description="Sliding window size for analysis",
    )
    stride: int = Field(default=64, ge=1, le=4096, description="Stride between windows")
    request_id: Optional[str] = Field(default=None, max_length=128)


# Tool 2: Manifold Audit

class ManifoldAuditInput(BaseModel):
    """Full spectral manifold analysis from hidden states or eigenvalues."""

    model_config = ConfigDict(extra="forbid")

    hidden_states: Optional[List[List[float]]] = Field(
        default=None,
        description="2D matrix [tokens x dimensions]. Provide this OR eigenvalues.",
    )
    eigenvalues: Optional[List[float]] = Field(
        default=None,
        description="Precomputed eigenvalue spectrum. Provide this OR hidden_states.",
    )
    source_type: Literal["hidden_states", "eigenvalues"] = Field(
        ...,
        description="Which input you are providing",
    )
    normalize: bool = Field(default=True, description="Normalize hidden states before analysis")
    center: bool = Field(default=True, description="Center hidden states (subtract mean)")
    return_eigenvalues: bool = Field(default=False, description="Include raw eigenvalues in response")
    request_id: Optional[str] = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def check_source(self):
        if self.source_type == "hidden_states" and not self.hidden_states:
            raise ValueError("hidden_states required when source_type='hidden_states'")
        if self.source_type == "eigenvalues" and not self.eigenvalues:
            raise ValueError("eigenvalues required when source_type='eigenvalues'")

        # Size guards
        if self.hidden_states:
            if len(self.hidden_states) > 8192:
                raise ValueError(f"Max 8192 samples, got {len(self.hidden_states)}")
            if self.hidden_states and len(self.hidden_states[0]) > 4096:
                raise ValueError(f"Max 4096 dimensions, got {len(self.hidden_states[0])}")
        if self.eigenvalues and len(self.eigenvalues) > 65536:
            raise ValueError(f"Max 65536 eigenvalues, got {len(self.eigenvalues)}")
        return self


# Tool 3: Compute Correction

class CorrectionInput(BaseModel):
    """Compute intervention signal to restore GUE rigidity."""

    model_config = ConfigDict(extra="forbid")

    current_r_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current measured spacing ratio <r>",
    )
    target_r_ratio: float = Field(
        default=0.578,
        ge=0.0,
        le=1.0,
        description="Target spacing ratio (default: GUE attractor 0.578)",
    )
    gain: float = Field(default=1.0, ge=0.0, le=10.0, description="Correction gain multiplier")
    clamp_output: bool = Field(default=True, description="Clamp intervention signal magnitude")
    max_magnitude: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Max correction magnitude",
    )
    request_id: Optional[str] = Field(default=None, max_length=128)


# Tool 4: Compare Models

class ModelSpecInput(BaseModel):
    """Spectral data for one model in a comparison."""

    model_config = ConfigDict(extra="forbid")

    model_label: str = Field(..., min_length=1, max_length=128, description="Label for this model")
    hidden_states: Optional[List[List[float]]] = None
    eigenvalues: Optional[List[float]] = None
    source_type: Literal["hidden_states", "eigenvalues"] = "hidden_states"


class CompareInput(BaseModel):
    """Compare spectral health between two models/checkpoints."""

    model_config = ConfigDict(extra="forbid")

    left: ModelSpecInput
    right: ModelSpecInput
    normalize: bool = Field(default=True)
    request_id: Optional[str] = Field(default=None, max_length=128)


# ==============================================
# MCP Tools
# ==============================================


@mcp.tool(
    name="brain_health_check",
    annotations=_tool_annotations("Spectral Health Check"),
)
async def brain_health_check(params: HealthCheckInput) -> str:
    """Analyze text for spectral health using GUE spacing ratio.

    Performs sliding-window analysis on input text to estimate the
    eigenvalue spacing ratio <r>. Values near 0.578 indicate GUE-like
    spectral rigidity (coherent reasoning). Values near 0.386 indicate
    Poisson-like spacing (context decoherence).

    This is a lightweight proxy measurement. For direct manifold analysis
    from model hidden states, use brain_manifold_audit instead.

    Returns spectral health score (0-100), estimated r-ratio, regime
    classification, confidence level, and drift warnings.
    """
    try:
        result = spectral_health_check(
            text=params.text,
            window_size=params.window_size,
            stride=params.stride,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = params.request_id
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "detail": str(e),
                "request_id": params.request_id,
            }
        )


@mcp.tool(
    name="brain_manifold_audit",
    annotations=_tool_annotations("Manifold Spectral Audit"),
)
async def brain_manifold_audit(params: ManifoldAuditInput) -> str:
    """Full spectral manifold analysis from model hidden states or eigenvalues.

    Computes the Gram matrix of hidden state vectors, extracts eigenvalues,
    and measures GUE spectral rigidity. Returns the spacing ratio <r>,
    spectral gap lambda_2, manifold coherence zeta, regime classification,
    and confidence metrics.

    Accepts either raw hidden state vectors [tokens x dimensions] or
    precomputed eigenvalue spectra.

    This is the core diagnostic tool for transformer latent space health.
    """
    try:
        result = manifold_audit(
            hidden_states=params.hidden_states,
            eigenvalues=params.eigenvalues,
            normalize=params.normalize,
            center=params.center,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = params.request_id
        if not params.return_eigenvalues:
            result.pop("eigenvalues", None)
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "detail": str(e),
                "request_id": params.request_id,
            }
        )


@mcp.tool(
    name="brain_compute_correction",
    annotations=_tool_annotations("Spectral Correction Signal"),
)
async def brain_compute_correction(params: CorrectionInput) -> str:
    """Compute intervention signal to restore GUE spectral rigidity.

    Given the current measured spacing ratio and target (default 0.578),
    returns the correction delta, recommended sigma adjustment for the
    heat kernel, intervention direction, and a human-readable action.

    Use after brain_health_check or brain_manifold_audit to determine
    what adjustment is needed when a model drifts from rigidity.
    """
    try:
        result = compute_correction(
            current_r_ratio=params.current_r_ratio,
            target_r_ratio=params.target_r_ratio,
            gain=params.gain,
            clamp=params.clamp_output,
            max_magnitude=params.max_magnitude,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = params.request_id
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "detail": str(e),
                "request_id": params.request_id,
            }
        )


@mcp.tool(
    name="brain_compare_models",
    annotations=_tool_annotations("Compare Model Spectral Health"),
)
async def brain_compare_models(params: CompareInput) -> str:
    """Compare spectral health between two models, checkpoints, or layers.

    Runs manifold_audit on both inputs and returns a side-by-side
    comparison with delta scores, regime classifications, and a
    determination of which model is spectrally healthier.

    Use for: before/after intervention, checkpoint A vs B,
    model comparison, or prompt strategy evaluation.

    Returns comparability warnings when inputs differ significantly
    in size or preprocessing.
    """
    try:
        result = compare_models(
            left_hidden_states=params.left.hidden_states,
            left_eigenvalues=params.left.eigenvalues,
            right_hidden_states=params.right.hidden_states,
            right_eigenvalues=params.right.eigenvalues,
            normalize=params.normalize,
        )
        result["schema_version"] = SCHEMA_VERSION
        result["request_id"] = params.request_id
        result["left_label"] = params.left.model_label
        result["right_label"] = params.right.model_label
        return json.dumps(result, indent=2)
    except ValueError as e:
        return json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "detail": str(e),
                "request_id": params.request_id,
            }
        )


# ==============================================
# Entry Point
# ==============================================

if __name__ == "__main__":
    import sys

    # Default: stdio for local MCP clients (Claude Desktop, etc.)
    # Use --http for remote server mode.
    if "--http" in sys.argv:
        port = 8000
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        print(f"Starting Geometric Brain MCP on HTTP port {port}")
        run_kwargs: dict[str, Any] = {"transport": "streamable_http", "port": port}
        mcp.run(**run_kwargs)
    else:
        mcp.run()
