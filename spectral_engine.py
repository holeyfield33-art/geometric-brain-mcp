"""
Geometric Brain - Spectral Engine
Core computation layer. Pure NumPy. No torch dependency.

Merged from unitarity-lab core modules:
  - core/metrics.py (spacing ratio, SHI)
  - core/horizons.py (Lanczos spectral analysis)
  - core/bridge.py (correction/intervention signals)

Built through TMRP: Gemini (core functions) + ChatGPT (edge case hardening) + Claude (merge)
(c) 2026 Aletheia Sovereign Systems - MIT License
"""

from typing import Optional

import numpy as np

# GUE and Poisson constants (locked)
GUE_R = 0.578
POISSON_R = 0.386
MIN_SAMPLES_FOR_CONFIDENCE = 10
MIN_EIGENVALUES = 3


def _validate_finite(arr: np.ndarray, name: str) -> None:
    """Reject NaN/Inf before they poison the pipeline."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")


def _compute_r_ratios(spacings: np.ndarray) -> np.ndarray:
    """Compute adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})."""
    ratios = []
    for i in range(1, len(spacings)):
        s1, s2 = spacings[i - 1], spacings[i]
        denom = max(s1, s2)
        if denom > 0:
            ratios.append(min(s1, s2) / denom)
    return np.array(ratios) if ratios else np.array([])


def spectral_health_check(
    text: str,
    window_size: int = 128,
    stride: int = 64,
) -> dict:
    """
    Text in, spectral health score out.
    Uses token-level character encoding to build a heat kernel graph
    and estimate the spacing ratio <r>.

    This is a proxy measurement -- not direct manifold access.
    For true hidden-state analysis, use manifold_audit().
    """
    if not text or len(text) < 4:
        return {
            "status": "error",
            "error_code": "INSUFFICIENT_DATA",
            "r_ratio": None,
            "shi_score": 0.0,
            "drift_warning": False,
            "confidence": 0.0,
            "windows_analyzed": 0,
        }

    tokens = np.array([ord(c) for c in text], dtype=np.float64)
    _validate_finite(tokens, "tokens")

    # Sliding window analysis for stability
    all_r_ratios = []
    windows_analyzed = 0

    for start in range(0, len(tokens) - window_size + 1, stride):
        window = tokens[start : start + window_size]
        diffs = np.abs(np.diff(window))
        ratios = _compute_r_ratios(diffs)
        if len(ratios) > 0:
            all_r_ratios.extend(ratios.tolist())
            windows_analyzed += 1

    # If text is shorter than one window, analyze the whole thing
    if windows_analyzed == 0:
        diffs = np.abs(np.diff(tokens))
        ratios = _compute_r_ratios(diffs)
        if len(ratios) > 0:
            all_r_ratios.extend(ratios.tolist())
            windows_analyzed = 1

    if not all_r_ratios:
        return {
            "status": "error",
            "error_code": "NO_SPACINGS",
            "r_ratio": None,
            "shi_score": 0.0,
            "drift_warning": False,
            "confidence": 0.0,
            "windows_analyzed": 0,
        }

    avg_r = float(np.mean(all_r_ratios))
    gue_dist = abs(avg_r - GUE_R)
    poisson_dist = abs(avg_r - POISSON_R)

    # SHI score: 0-100 scale, 100 = perfect GUE rigidity
    shi_score = max(0.0, min(100.0, (1.0 - gue_dist) * 100.0))

    # Confidence scales with sample count
    n = len(all_r_ratios)
    confidence = min(1.0, n / 200.0)

    # Regime classification
    if gue_dist < 0.05:
        regime = "gue_like"
    elif poisson_dist < 0.05:
        regime = "poisson_like"
    else:
        regime = "intermediate"

    drift_warning = gue_dist > 0.1

    if gue_dist < poisson_dist:
        status = "ok"
    elif regime == "intermediate":
        status = "warning"
    else:
        status = "warning"

    return {
        "status": status,
        "error_code": None,
        "r_ratio": round(avg_r, 5),
        "shi_score": round(shi_score, 2),
        "regime": regime,
        "drift_warning": drift_warning,
        "confidence": round(confidence, 3),
        "windows_analyzed": windows_analyzed,
        "spacing_count": n,
    }


def manifold_audit(
    hidden_states: Optional[list] = None,
    eigenvalues: Optional[list] = None,
    normalize: bool = True,
    center: bool = True,
) -> dict:
    """
    Hidden state vectors (or precomputed eigenvalues) in,
    full manifold analysis out.

    Computes the Gram matrix, extracts eigenvalues via eigh,
    and determines spectral regime (GUE vs Poisson).
    """
    computed_eigenvalues = None

    if eigenvalues is not None:
        evals = np.array(eigenvalues, dtype=np.float64)
        _validate_finite(evals, "eigenvalues")
        evals = np.sort(evals)
        computed_eigenvalues = evals
        sample_count = len(evals)
        feature_count = 0

    elif hidden_states is not None:
        H = np.array(hidden_states, dtype=np.float64)
        if H.ndim != 2:
            return {
                "status": "error",
                "error_code": "INVALID_SHAPE",
                "summary": f"Expected 2D array, got {H.ndim}D",
            }

        _validate_finite(H, "hidden_states")
        sample_count, feature_count = H.shape

        if sample_count < MIN_EIGENVALUES:
            return {
                "status": "error",
                "error_code": "INSUFFICIENT_SAMPLES",
                "summary": f"Need at least {MIN_EIGENVALUES} samples, got {sample_count}",
            }

        # Preprocessing
        if center:
            H = H - H.mean(axis=0)
        if normalize:
            norms = np.linalg.norm(H, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            H = H / norms

        # Gram matrix -> eigenvalues
        G = np.dot(H, H.T)
        evals = np.linalg.eigvalsh(G)
        evals = np.sort(evals)
        computed_eigenvalues = evals

    else:
        return {
            "status": "error",
            "error_code": "NO_INPUT",
            "summary": "Provide either hidden_states or eigenvalues",
        }

    # Filter near-zero eigenvalues (numerical noise)
    threshold = np.max(np.abs(evals)) * 1e-10
    significant_evals = evals[np.abs(evals) > threshold]

    if len(significant_evals) < MIN_EIGENVALUES:
        return {
            "status": "warning",
            "error_code": "RANK_DEFICIENT",
            "summary": f"Only {len(significant_evals)} significant eigenvalues found",
            "spectral_health_score": 0.0,
            "confidence": 0.0,
        }

    # Spectral gap (lambda_2)
    lambda_2 = float(significant_evals[1] - significant_evals[0]) if len(significant_evals) > 1 else 0.0

    # Spacing ratios
    diffs = np.diff(significant_evals)
    r_ratios = _compute_r_ratios(diffs)

    if len(r_ratios) == 0:
        return {
            "status": "warning",
            "error_code": "NO_GAPS",
            "summary": "Could not compute spacing ratios",
            "spectral_health_score": 0.0,
            "confidence": 0.0,
        }

    avg_r = float(np.mean(r_ratios))
    var_r = float(np.var(r_ratios))
    gue_dist = abs(avg_r - GUE_R)
    poisson_dist = abs(avg_r - POISSON_R)

    # Regime
    if gue_dist < 0.05:
        regime = "gue_like"
    elif poisson_dist < 0.05:
        regime = "poisson_like"
    else:
        regime = "intermediate"

    # Zeta score (manifold coherence proxy)
    mean_eval = np.mean(significant_evals)
    zeta_score = float(np.std(significant_evals) / mean_eval) if abs(mean_eval) > 1e-10 else 0.0

    # SHI and confidence
    shi_score = max(0.0, min(100.0, (1.0 - gue_dist) * 100.0))
    confidence = min(1.0, len(r_ratios) / 50.0)

    summary_parts = [
        f"Regime: {regime}",
        f"<r> = {avg_r:.4f} (GUE target: {GUE_R})",
        f"Spectral gap lambda_2 = {lambda_2:.6f}",
        f"Based on {len(r_ratios)} spacing ratios from {len(significant_evals)} eigenvalues",
    ]

    return {
        "status": "ok" if regime == "gue_like" else "warning",
        "error_code": None,
        "sample_count": sample_count,
        "feature_count": feature_count,
        "spectral_health_score": round(shi_score, 2),
        "mean_r_ratio": round(avg_r, 5),
        "variance_r_ratio": round(var_r, 6),
        "spectral_regime": regime,
        "lambda_2": round(lambda_2, 8),
        "zeta_score": round(zeta_score, 6),
        "gue_distance": round(gue_dist, 5),
        "poisson_distance": round(poisson_dist, 5),
        "spacing_count": len(r_ratios),
        "confidence": round(confidence, 3),
        "eigenvalues": significant_evals.tolist(),
        "summary": " | ".join(summary_parts),
    }


def compute_correction(
    current_r_ratio: float,
    target_r_ratio: float = GUE_R,
    gain: float = 1.0,
    clamp: bool = True,
    max_magnitude: float = 1.0,
) -> dict:
    """
    Given current r-ratio and target, returns intervention signal
    to pull the manifold back toward GUE rigidity.
    """
    if not 0.0 <= current_r_ratio <= 1.0:
        return {
            "status": "error",
            "error_code": "INVALID_R_RATIO",
            "delta": 0.0,
        }

    error = target_r_ratio - current_r_ratio
    delta_k = error * np.pi * gain

    if clamp:
        delta_k = float(np.clip(delta_k, -max_magnitude, max_magnitude))

    magnitude = abs(delta_k)

    # Direction
    if abs(error) < 0.005:
        direction = "hold"
        action = "No intervention needed. Manifold is at target rigidity."
    elif error > 0:
        direction = "increase_repulsion"
        action = (
            f"Increase eigenvalue repulsion. Current <r>={current_r_ratio:.4f} "
            f"is below target {target_r_ratio:.4f}."
        )
    else:
        direction = "decrease_repulsion"
        action = (
            f"Decrease eigenvalue repulsion. Current <r>={current_r_ratio:.4f} "
            f"is above target {target_r_ratio:.4f}."
        )

    # Recommended sigma for heat kernel (k=1 invariant bridge)
    rec_sigma = 1.0 + error if error > 0 else 1.0 / (1.0 + abs(error))

    # Confidence: higher when closer to target
    confidence = max(0.0, min(1.0, 1.0 - magnitude))

    return {
        "status": "ok",
        "error_code": None,
        "current_r_ratio": round(current_r_ratio, 5),
        "target_r_ratio": round(target_r_ratio, 5),
        "delta": round(delta_k, 6),
        "intervention_signal": round(delta_k, 6),
        "magnitude": round(magnitude, 6),
        "direction": direction,
        "recommended_action": action,
        "recommended_sigma": round(rec_sigma, 4),
        "confidence": round(confidence, 3),
    }


def compare_models(
    left_hidden_states: Optional[list] = None,
    left_eigenvalues: Optional[list] = None,
    right_hidden_states: Optional[list] = None,
    right_eigenvalues: Optional[list] = None,
    normalize: bool = True,
) -> dict:
    """
    Compare spectral health of two models/checkpoints/layers.
    Premium tool: the decision layer.
    """
    left = manifold_audit(
        hidden_states=left_hidden_states,
        eigenvalues=left_eigenvalues,
        normalize=normalize,
    )
    right = manifold_audit(
        hidden_states=right_hidden_states,
        eigenvalues=right_eigenvalues,
        normalize=normalize,
    )

    # Handle errors from either side
    if left.get("status") == "error":
        return {"status": "error", "error_code": "LEFT_FAILED", "detail": left}
    if right.get("status") == "error":
        return {"status": "error", "error_code": "RIGHT_FAILED", "detail": right}

    left_score = left.get("spectral_health_score", 0.0)
    right_score = right.get("spectral_health_score", 0.0)
    left_r = left.get("mean_r_ratio")
    right_r = right.get("mean_r_ratio")

    delta_score = left_score - right_score
    delta_r = (left_r - right_r) if (left_r is not None and right_r is not None) else None

    if abs(delta_score) < 1.0:
        healthier = "tie"
    elif delta_score > 0:
        healthier = "left"
    else:
        healthier = "right"

    # Comparability warnings
    warnings = []
    left_n = left.get("spacing_count", 0)
    right_n = right.get("spacing_count", 0)
    if left_n > 0 and right_n > 0:
        ratio = max(left_n, right_n) / max(min(left_n, right_n), 1)
        if ratio > 5.0:
            warnings.append(f"Sample size imbalance: {left_n} vs {right_n} spacings")

    return {
        "status": "ok",
        "error_code": None,
        "left_health_score": left_score,
        "right_health_score": right_score,
        "left_r_ratio": left_r,
        "right_r_ratio": right_r,
        "left_regime": left.get("spectral_regime", "unknown"),
        "right_regime": right.get("spectral_regime", "unknown"),
        "healthier_model": healthier,
        "delta_health_score": round(delta_score, 2),
        "delta_r_ratio": round(delta_r, 5) if delta_r is not None else None,
        "comparative_summary": (
            f"{'Left' if healthier == 'left' else 'Right' if healthier == 'right' else 'Both models'} "
            f"{'is' if healthier != 'tie' else 'are equally'} spectrally "
            f"{'healthier' if healthier != 'tie' else 'healthy'} (delta SHI: {abs(delta_score):.1f})"
        ),
        "warnings": warnings,
    }
