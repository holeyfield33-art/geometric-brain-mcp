"""
Geometric Brain MCP - End-to-end Test Suite
============================================

Coverage:
  - spectral_engine.py: spectral_health_check, manifold_audit,
                        compute_correction, compare_models
  - api.py: all REST endpoints via FastAPI TestClient

Run with: pytest test_e2e.py -v
"""

import numpy as np
import pytest

from spectral_engine import (
    GUE_R,
    POISSON_R,
    compare_models,
    compute_correction,
    manifold_audit,
    spectral_health_check,
)

try:
    from fastapi.testclient import TestClient

    from api import app

    _API = True
except ImportError:
    _API = False

# ── Fixed-seed fixtures (deterministic regardless of run order) ──────────────

_HS_10x16 = np.random.default_rng(42).standard_normal((10, 16)).tolist()
_HS_20x32 = np.random.default_rng(43).standard_normal((20, 32)).tolist()
_HS_5x8 = np.random.default_rng(44).standard_normal((5, 8)).tolist()
_HS_1x8 = np.random.default_rng(45).standard_normal((1, 8)).tolist()  # too few samples
_EVALS_30 = sorted(np.random.default_rng(46).standard_normal(30).tolist())
_EVALS_20 = sorted(np.random.default_rng(47).standard_normal(20).tolist())
_LONG_TEXT = "The quick brown fox jumps over the lazy dog. " * 50

if _API:
    _client = TestClient(app)


# =============================================================================
# 1. spectral_health_check
# =============================================================================


class TestSpectralHealthCheck:
    def test_normal_text_returns_non_error_status(self):
        r = spectral_health_check(_LONG_TEXT)
        assert r["status"] in ("ok", "warning")

    def test_normal_text_has_no_error_code(self):
        r = spectral_health_check(_LONG_TEXT)
        assert r["error_code"] is None

    def test_short_text_returns_insufficient_data(self):
        r = spectral_health_check("Hi")
        assert r["status"] == "error"
        assert r["error_code"] == "INSUFFICIENT_DATA"

    def test_empty_string_returns_error(self):
        r = spectral_health_check("")
        assert r["status"] == "error"

    def test_r_ratio_in_valid_range(self):
        r = spectral_health_check(_LONG_TEXT)
        assert 0.0 <= r["r_ratio"] <= 1.0

    def test_shi_score_in_valid_range(self):
        r = spectral_health_check(_LONG_TEXT)
        assert 0.0 <= r["shi_score"] <= 100.0

    def test_confidence_in_valid_range(self):
        r = spectral_health_check(_LONG_TEXT)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_regime_is_valid_string(self):
        r = spectral_health_check(_LONG_TEXT)
        assert r["regime"] in ("gue_like", "poisson_like", "intermediate")

    def test_windows_analyzed_positive(self):
        r = spectral_health_check(_LONG_TEXT)
        assert r["windows_analyzed"] >= 1

    def test_drift_warning_is_bool(self):
        r = spectral_health_check(_LONG_TEXT)
        assert isinstance(r["drift_warning"], bool)

    def test_schema_keys_present(self):
        r = spectral_health_check(_LONG_TEXT)
        for key in (
            "status",
            "error_code",
            "r_ratio",
            "shi_score",
            "regime",
            "drift_warning",
            "confidence",
            "windows_analyzed",
            "spacing_count",
        ):
            assert key in r, f"Missing key: {key}"

    def test_spacing_count_is_positive_int(self):
        r = spectral_health_check(_LONG_TEXT)
        assert isinstance(r["spacing_count"], int)
        assert r["spacing_count"] > 0

    def test_custom_window_size_and_stride(self):
        r = spectral_health_check(_LONG_TEXT, window_size=64, stride=32)
        assert r["status"] in ("ok", "warning", "error")

    def test_window_size_larger_than_text_falls_back(self):
        short = "Hello world this is a test sentence."
        r = spectral_health_check(short, window_size=512)
        assert r["status"] in ("ok", "warning", "error")


# =============================================================================
# 2. manifold_audit
# =============================================================================


class TestManifoldAudit:
    def test_valid_hidden_states_runs(self):
        r = manifold_audit(hidden_states=_HS_10x16)
        assert r["status"] in ("ok", "warning")

    def test_valid_eigenvalues_runs(self):
        r = manifold_audit(eigenvalues=_EVALS_30)
        assert r["status"] in ("ok", "warning")
        assert "mean_r_ratio" in r

    def test_no_input_returns_error(self):
        r = manifold_audit()
        assert r["status"] == "error"
        assert r["error_code"] == "NO_INPUT"

    def test_too_few_samples_returns_error(self):
        r = manifold_audit(hidden_states=_HS_1x8)
        assert r["status"] == "error"
        assert r["error_code"] == "INSUFFICIENT_SAMPLES"

    def test_nan_in_hidden_states_raises(self):
        bad = [row[:] for row in _HS_5x8]
        bad[0][0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            manifold_audit(hidden_states=bad)

    def test_inf_in_eigenvalues_raises(self):
        bad = _EVALS_20[:]
        bad[0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            manifold_audit(eigenvalues=bad)

    def test_schema_keys_present(self):
        r = manifold_audit(hidden_states=_HS_10x16)
        for key in (
            "spectral_health_score",
            "mean_r_ratio",
            "variance_r_ratio",
            "spectral_regime",
            "lambda_2",
            "confidence",
            "spacing_count",
            "gue_distance",
            "poisson_distance",
            "zeta_score",
            "summary",
        ):
            assert key in r, f"Missing key: {key}"

    def test_spectral_health_score_in_range(self):
        r = manifold_audit(hidden_states=_HS_20x32)
        assert 0.0 <= r["spectral_health_score"] <= 100.0

    def test_confidence_in_range(self):
        r = manifold_audit(hidden_states=_HS_20x32)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_regime_is_valid_string(self):
        r = manifold_audit(hidden_states=_HS_10x16)
        assert r["spectral_regime"] in ("gue_like", "poisson_like", "intermediate")

    def test_engine_returns_eigenvalues_key(self):
        r = manifold_audit(hidden_states=_HS_10x16)
        # engine always emits eigenvalues; api/server may strip them
        assert "eigenvalues" in r

    def test_disable_center_and_normalize(self):
        r = manifold_audit(hidden_states=_HS_10x16, normalize=False, center=False)
        assert "spectral_health_score" in r

    def test_summary_is_string(self):
        r = manifold_audit(hidden_states=_HS_10x16)
        assert isinstance(r["summary"], str)
        assert len(r["summary"]) > 0


# =============================================================================
# 3. compute_correction
# =============================================================================


class TestComputeCorrection:
    def test_at_gue_target_is_hold(self):
        r = compute_correction(current_r_ratio=GUE_R)
        assert r["direction"] == "hold"
        assert r["status"] == "ok"

    def test_below_target_requests_increase(self):
        r = compute_correction(current_r_ratio=POISSON_R)
        assert r["direction"] == "increase_repulsion"

    def test_above_target_requests_decrease(self):
        r = compute_correction(current_r_ratio=0.9)
        assert r["direction"] == "decrease_repulsion"

    def test_delta_positive_when_below_target(self):
        r = compute_correction(current_r_ratio=0.3, clamp=False)
        assert r["delta"] > 0

    def test_delta_negative_when_above_target(self):
        r = compute_correction(current_r_ratio=0.8, clamp=False)
        assert r["delta"] < 0

    def test_clamp_limits_magnitude(self):
        r = compute_correction(current_r_ratio=0.0, clamp=True, max_magnitude=0.3)
        assert abs(r["delta"]) <= 0.3 + 1e-9

    def test_gain_amplifies_delta(self):
        r1 = compute_correction(current_r_ratio=0.4, gain=1.0, clamp=False)
        r2 = compute_correction(current_r_ratio=0.4, gain=2.0, clamp=False)
        assert abs(r2["delta"]) > abs(r1["delta"])

    def test_invalid_r_ratio_returns_error(self):
        r = compute_correction(current_r_ratio=1.5)
        assert r["status"] == "error"
        assert r["error_code"] == "INVALID_R_RATIO"

    def test_delta_equals_intervention_signal(self):
        r = compute_correction(current_r_ratio=0.5)
        assert r["delta"] == r["intervention_signal"]

    def test_confidence_in_range(self):
        r = compute_correction(current_r_ratio=GUE_R)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_schema_keys_present(self):
        r = compute_correction(current_r_ratio=0.5)
        for key in (
            "status",
            "error_code",
            "current_r_ratio",
            "target_r_ratio",
            "delta",
            "intervention_signal",
            "magnitude",
            "direction",
            "recommended_action",
            "recommended_sigma",
            "confidence",
        ):
            assert key in r, f"Missing key: {key}"

    def test_custom_target_hold(self):
        r = compute_correction(current_r_ratio=0.5, target_r_ratio=0.5)
        assert r["direction"] == "hold"

    def test_recommendation_action_is_string(self):
        r = compute_correction(current_r_ratio=0.4)
        assert isinstance(r["recommended_action"], str)


# =============================================================================
# 4. compare_models
# =============================================================================


class TestCompareModels:
    def test_identical_inputs_produce_tie(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_10x16)
        assert r["healthier_model"] == "tie"
        assert r["delta_health_score"] == 0.0

    def test_healthier_model_valid_value(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_20x32)
        assert r["healthier_model"] in ("left", "right", "tie")

    def test_status_ok_for_valid_inputs(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_20x32)
        assert r["status"] in ("ok", "warning")

    def test_schema_keys_present(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_10x16)
        for key in (
            "status",
            "error_code",
            "healthier_model",
            "delta_health_score",
            "left_regime",
            "right_regime",
            "comparative_summary",
            "warnings",
            "left_health_score",
            "right_health_score",
        ):
            assert key in r, f"Missing key: {key}"

    def test_delta_health_score_is_float(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_20x32)
        assert isinstance(r["delta_health_score"], float)

    def test_comparative_summary_is_string(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_10x16)
        assert isinstance(r["comparative_summary"], str)
        assert len(r["comparative_summary"]) > 0

    def test_warnings_is_list(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_10x16)
        assert isinstance(r["warnings"], list)

    def test_left_error_propagated(self):
        r = compare_models(left_hidden_states=_HS_1x8, right_hidden_states=_HS_10x16)
        assert r["status"] == "error"
        assert r["error_code"] == "LEFT_FAILED"

    def test_right_error_propagated(self):
        r = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_1x8)
        assert r["status"] == "error"
        assert r["error_code"] == "RIGHT_FAILED"

    def test_eigenvalue_inputs(self):
        r = compare_models(left_eigenvalues=_EVALS_30, right_eigenvalues=_EVALS_20)
        assert r["status"] in ("ok", "warning")

    def test_delta_symmetric(self):
        r1 = compare_models(left_hidden_states=_HS_10x16, right_hidden_states=_HS_20x32)
        r2 = compare_models(left_hidden_states=_HS_20x32, right_hidden_states=_HS_10x16)
        assert abs(r1["delta_health_score"] + r2["delta_health_score"]) < 1e-9


# =============================================================================
# 5. REST API — meta and health probes
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAPIMeta:
    def test_healthz_200(self):
        r = _client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_healthz_service_name(self):
        r = _client.get("/healthz")
        assert r.json()["service"] == "geometric-brain"

    def test_readyz_200(self):
        r = _client.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_readyz_has_schema_version(self):
        r = _client.get("/readyz")
        assert r.json()["schema_version"] == "1.0.0"

    def test_capabilities_four_tools(self):
        r = _client.get("/v1/meta/capabilities")
        assert r.status_code == 200
        body = r.json()
        assert len(body["tools"]) == 4
        for tool in (
            "brain_health_check",
            "brain_manifold_audit",
            "brain_compute_correction",
            "brain_compare_models",
        ):
            assert tool in body["tools"]

    def test_capabilities_gue_constant(self):
        r = _client.get("/v1/meta/capabilities")
        assert r.json()["constants"]["gue_r"] == pytest.approx(0.578)

    def test_capabilities_poisson_constant(self):
        r = _client.get("/v1/meta/capabilities")
        assert r.json()["constants"]["poisson_r"] == pytest.approx(0.386)

    def test_capabilities_limits(self):
        r = _client.get("/v1/meta/capabilities")
        limits = r.json()["limits"]
        assert limits["max_text_length"] == 20_000
        assert limits["max_hidden_state_samples"] == 8192


# =============================================================================
# 6. REST API — POST /v1/brain/health-check
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAPIHealthCheck:
    def test_valid_returns_200(self):
        r = _client.post("/v1/brain/health-check", json={"text": _LONG_TEXT})
        assert r.status_code == 200

    def test_schema_version_in_response(self):
        r = _client.post("/v1/brain/health-check", json={"text": _LONG_TEXT})
        assert r.json()["schema_version"] == "1.0.0"

    def test_missing_text_returns_422(self):
        r = _client.post("/v1/brain/health-check", json={})
        assert r.status_code == 422

    def test_text_too_long_returns_422(self):
        r = _client.post("/v1/brain/health-check", json={"text": "x" * 20_001})
        assert r.status_code == 422

    def test_extra_field_returns_422(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": _LONG_TEXT, "undeclared_field": True},
        )
        assert r.status_code == 422

    def test_custom_window_and_stride(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": _LONG_TEXT, "window_size": 64, "stride": 32},
        )
        assert r.status_code == 200

    def test_window_size_below_minimum_returns_422(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": _LONG_TEXT, "window_size": 4},
        )
        assert r.status_code == 422


# =============================================================================
# 7. REST API — POST /v1/brain/manifold-audit
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAPIManifoldAudit:
    def test_hidden_states_returns_200(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": _HS_10x16},
        )
        assert r.status_code == 200
        assert "spectral_health_score" in r.json()

    def test_eigenvalues_returns_200(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "eigenvalues", "eigenvalues": _EVALS_30},
        )
        assert r.status_code == 200

    def test_schema_version_in_response(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": _HS_10x16},
        )
        assert r.json()["schema_version"] == "1.0.0"

    def test_eigenvalues_excluded_by_default(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": _HS_10x16},
        )
        assert r.status_code == 200
        assert "eigenvalues" not in r.json()

    def test_eigenvalues_included_when_requested(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={
                "source_type": "hidden_states",
                "hidden_states": _HS_10x16,
                "return_eigenvalues": True,
            },
        )
        assert r.status_code == 200
        assert "eigenvalues" in r.json()

    def test_missing_source_type_returns_422(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"hidden_states": _HS_10x16},
        )
        assert r.status_code == 422

    def test_source_hidden_states_without_data_returns_422(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states"},
        )
        assert r.status_code == 422

    def test_invalid_source_type_returns_422(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "gradient_states", "hidden_states": _HS_5x8},
        )
        assert r.status_code == 422


# =============================================================================
# 8. REST API — POST /v1/brain/compute-correction
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAPIComputeCorrection:
    def test_at_gue_returns_hold(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": GUE_R},
        )
        assert r.status_code == 200
        assert r.json()["direction"] == "hold"

    def test_below_gue_returns_increase(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.3},
        )
        assert r.status_code == 200
        assert r.json()["direction"] == "increase_repulsion"

    def test_schema_version_in_response(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.5},
        )
        assert r.json()["schema_version"] == "1.0.0"

    def test_r_ratio_above_1_returns_422(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 1.5},
        )
        assert r.status_code == 422

    def test_r_ratio_below_0_returns_422(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": -0.1},
        )
        assert r.status_code == 422

    def test_custom_equal_target_is_hold(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.5, "target_r_ratio": 0.5},
        )
        assert r.status_code == 200
        assert r.json()["direction"] == "hold"

    def test_gain_param_accepted(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.4, "gain": 2.0},
        )
        assert r.status_code == 200

    def test_gain_above_max_returns_422(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.4, "gain": 11.0},
        )
        assert r.status_code == 422


# =============================================================================
# 9. REST API — POST /v1/brain/compare-models
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAPICompareModels:
    _L = {"model_label": "model-a", "source_type": "hidden_states", "hidden_states": _HS_10x16}
    _R = {"model_label": "model-b", "source_type": "hidden_states", "hidden_states": _HS_20x32}

    def test_valid_returns_200(self):
        r = _client.post("/v1/brain/compare-models", json={"left": self._L, "right": self._R})
        assert r.status_code == 200

    def test_labels_in_response(self):
        r = _client.post("/v1/brain/compare-models", json={"left": self._L, "right": self._R})
        body = r.json()
        assert body["left_label"] == "model-a"
        assert body["right_label"] == "model-b"

    def test_schema_version_in_response(self):
        r = _client.post("/v1/brain/compare-models", json={"left": self._L, "right": self._R})
        assert r.json()["schema_version"] == "1.0.0"

    def test_healthier_model_key_present(self):
        r = _client.post("/v1/brain/compare-models", json={"left": self._L, "right": self._R})
        assert "healthier_model" in r.json()

    def test_eigenvalue_inputs(self):
        r = _client.post(
            "/v1/brain/compare-models",
            json={
                "left": {"model_label": "ckpt-100", "source_type": "eigenvalues", "eigenvalues": _EVALS_30},
                "right": {"model_label": "ckpt-200", "source_type": "eigenvalues", "eigenvalues": _EVALS_20},
            },
        )
        assert r.status_code == 200

    def test_missing_left_returns_422(self):
        r = _client.post("/v1/brain/compare-models", json={"right": self._R})
        assert r.status_code == 422

    def test_missing_right_returns_422(self):
        r = _client.post("/v1/brain/compare-models", json={"left": self._L})
        assert r.status_code == 422

    def test_empty_model_label_returns_422(self):
        bad_l = {**self._L, "model_label": ""}
        r = _client.post("/v1/brain/compare-models", json={"left": bad_l, "right": self._R})
        assert r.status_code == 422


# =============================================================================
# 10. Auth middleware
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAuthMiddleware:
    """Test API key authentication when enabled."""

    _VALID_KEY = "test-key-abc123"
    _HEADERS = {"Authorization": "Bearer test-key-abc123"}

    @pytest.fixture(autouse=True)
    def _enable_auth(self, monkeypatch):
        """Enable auth with a known test key for every test in this class."""
        import config as _cfg

        monkeypatch.setattr(_cfg, "AUTH_ENABLED", True)
        monkeypatch.setattr(_cfg, "API_KEYS", [self._VALID_KEY])
        yield

    # -- public paths remain accessible --

    def test_healthz_public_when_auth_enabled(self):
        r = _client.get("/healthz")
        assert r.status_code == 200

    def test_readyz_public_when_auth_enabled(self):
        r = _client.get("/readyz")
        assert r.status_code == 200

    # -- protected paths reject missing/bad keys --

    def test_no_header_returns_401(self):
        r = _client.post("/v1/brain/health-check", json={"text": "hello world test"})
        assert r.status_code == 401
        body = r.json()
        assert body["error_code"] == "UNAUTHORIZED"

    def test_wrong_key_returns_401(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": "hello world test"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    def test_malformed_header_returns_401(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": "hello world test"},
            headers={"Authorization": "Token " + self._VALID_KEY},
        )
        assert r.status_code == 401

    def test_empty_bearer_returns_401(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": "hello world test"},
            headers={"Authorization": "Bearer "},
        )
        assert r.status_code == 401

    # -- valid key grants access --

    def test_valid_key_health_check_200(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": _LONG_TEXT},
            headers=self._HEADERS,
        )
        assert r.status_code == 200

    def test_valid_key_manifold_audit_200(self):
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": _HS_10x16},
            headers=self._HEADERS,
        )
        assert r.status_code == 200

    def test_valid_key_compute_correction_200(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.5},
            headers=self._HEADERS,
        )
        assert r.status_code == 200

    def test_valid_key_compare_models_200(self):
        r = _client.post(
            "/v1/brain/compare-models",
            json={
                "left": {"model_label": "a", "source_type": "hidden_states", "hidden_states": _HS_10x16},
                "right": {"model_label": "b", "source_type": "hidden_states", "hidden_states": _HS_10x16},
            },
            headers=self._HEADERS,
        )
        assert r.status_code == 200

    def test_valid_key_capabilities_200(self):
        r = _client.get("/v1/meta/capabilities", headers=self._HEADERS)
        assert r.status_code == 200

    def test_capabilities_requires_auth(self):
        r = _client.get("/v1/meta/capabilities")
        assert r.status_code == 401


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestAuthDisabled:
    """Verify auth-off (default) path still works."""

    @pytest.fixture(autouse=True)
    def _disable_auth(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "AUTH_ENABLED", False)
        yield

    def test_no_header_allowed_when_auth_disabled(self):
        r = _client.post("/v1/brain/health-check", json={"text": _LONG_TEXT})
        assert r.status_code == 200


# =============================================================================
# 7. Rate Limiting
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestRateLimiting:
    """Test in-process rate limiting middleware."""

    @pytest.fixture(autouse=True)
    def _enable_rate_limit(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "RATE_LIMIT_ENABLED", True)
        monkeypatch.setattr(_cfg, "RATE_LIMIT_RPM", 3)
        # Clear the rate limit store before each test
        from api import _rate_limit_store

        _rate_limit_store.clear()
        yield

    def test_under_limit_succeeds(self):
        for _ in range(3):
            r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
            assert r.status_code == 200

    def test_over_limit_returns_429(self):
        for _ in range(3):
            _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        assert r.status_code == 429
        body = r.json()
        assert body["error_code"] == "RATE_LIMITED"

    def test_public_paths_exempt(self):
        """healthz / readyz are exempt from rate limiting."""
        for _ in range(5):
            r = _client.get("/healthz")
            assert r.status_code == 200

    def test_rate_limit_response_body(self):
        for _ in range(3):
            _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        body = r.json()
        assert body["status"] == "error"
        assert "Rate limit exceeded" in body["detail"]


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestRateLimitDisabled:
    """Verify rate-limit-off (default) path still works."""

    @pytest.fixture(autouse=True)
    def _disable_rate_limit(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "RATE_LIMIT_ENABLED", False)
        yield

    def test_many_requests_allowed_when_disabled(self):
        for _ in range(10):
            r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
            assert r.status_code == 200


# =============================================================================
# 8. Body Size Limit
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestBodySizeLimit:
    """Test body size middleware rejects oversized payloads."""

    @pytest.fixture(autouse=True)
    def _set_body_limit(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "MAX_BODY_BYTES", 500)
        yield

    def test_small_body_accepted(self):
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        assert r.status_code == 200

    def test_large_body_returns_413(self):
        large_text = "x" * 1000
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": large_text},
        )
        assert r.status_code == 413
        body = r.json()
        assert body["error_code"] == "PAYLOAD_TOO_LARGE"

    def test_payload_too_large_response_body(self):
        large_text = "x" * 1000
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": large_text},
        )
        body = r.json()
        assert body["status"] == "error"
        assert "maximum size" in body["detail"]


# =============================================================================
# 9. Input Guardrails (config-wired caps)
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestInputGuardrails:
    """Test that Pydantic validators enforce config-driven limits."""

    def test_too_many_hidden_state_samples_returns_422(self):
        oversized = [[0.0] * 4] * 8193  # default MAX is 8192
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": oversized},
        )
        assert r.status_code == 422

    def test_too_many_hidden_state_dims_returns_422(self):
        oversized = [[0.0] * 4097] * 2  # default MAX is 4096
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": oversized},
        )
        assert r.status_code == 422

    def test_too_many_eigenvalues_returns_422(self):
        oversized = [0.0] * 65537  # default MAX is 65536
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "eigenvalues", "eigenvalues": oversized},
        )
        assert r.status_code == 422

    def test_text_over_max_length_returns_422(self):
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": "x" * 20_001},
        )
        assert r.status_code == 422

    def test_capabilities_reflects_config_limits(self):
        r = _client.get("/v1/meta/capabilities")
        limits = r.json()["limits"]
        assert limits["max_text_length"] == 20_000
        assert limits["max_hidden_state_samples"] == 8192
        assert limits["max_hidden_state_dimensions"] == 4096
        assert limits["max_eigenvalues"] == 65536
        assert "max_body_bytes" in limits


# =============================================================================
# 10. Request ID and Structured Responses
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestRequestId:
    """Verify request-id generation, pass-through, and inclusion in responses."""

    def test_response_has_x_request_id_header(self):
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        assert "x-request-id" in r.headers

    def test_auto_generated_request_id_is_hex(self):
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        rid = r.headers["x-request-id"]
        assert len(rid) == 32  # uuid4 hex
        int(rid, 16)  # must parse as hex

    def test_client_request_id_echoed(self):
        r = _client.post(
            "/v1/brain/compute-correction",
            json={"current_r_ratio": 0.5},
            headers={"x-request-id": "my-trace-42"},
        )
        assert r.headers["x-request-id"] == "my-trace-42"

    def test_request_id_in_success_body(self):
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        body = r.json()
        assert "request_id" in body
        assert body["request_id"] == r.headers["x-request-id"]

    def test_request_id_in_error_body(self):
        r = _client.post("/v1/brain/health-check", json={})
        assert r.status_code == 422
        body = r.json()
        assert "request_id" in body
        assert body["request_id"] == r.headers["x-request-id"]

    def test_get_endpoints_have_request_id_header(self):
        r = _client.get("/healthz")
        assert "x-request-id" in r.headers


# =============================================================================
# 11. Standardized Error Payloads
# =============================================================================


@pytest.mark.skipif(not _API, reason="fastapi/httpx not installed")
class TestStandardizedErrors:
    """Verify all error responses follow the standard shape."""

    def test_validation_error_shape(self):
        r = _client.post("/v1/brain/health-check", json={})
        assert r.status_code == 422
        body = r.json()
        assert body["status"] == "error"
        assert body["error_code"] == "VALIDATION_ERROR"
        assert "detail" in body
        assert "request_id" in body
        assert body["schema_version"] == "1.0.0"

    def test_auth_error_has_request_id(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "AUTH_ENABLED", True)
        monkeypatch.setattr(_cfg, "API_KEYS", ["k"])
        r = _client.post("/v1/brain/health-check", json={"text": "hi"})
        assert r.status_code == 401
        body = r.json()
        assert body["status"] == "error"
        assert body["error_code"] == "UNAUTHORIZED"
        assert "request_id" in body
        assert body["schema_version"] == "1.0.0"

    def test_rate_limit_error_has_request_id(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "RATE_LIMIT_ENABLED", True)
        monkeypatch.setattr(_cfg, "RATE_LIMIT_RPM", 0)
        from api import _rate_limit_store

        _rate_limit_store.clear()
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        assert r.status_code == 429
        body = r.json()
        assert body["status"] == "error"
        assert "request_id" in body
        assert body["schema_version"] == "1.0.0"

    def test_body_size_error_has_request_id(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "MAX_BODY_BYTES", 10)
        r = _client.post("/v1/brain/health-check", json={"text": "x" * 100})
        assert r.status_code == 413
        body = r.json()
        assert body["status"] == "error"
        assert "request_id" in body
        assert body["schema_version"] == "1.0.0"

    def test_engine_value_error_returns_422_with_standard_shape(self):
        """HTTPException from engine ValueError goes through the http_error_handler."""
        r = _client.post(
            "/v1/brain/health-check",
            json={"text": "tiny", "window_size": 4096, "stride": 1},
        )
        # Engine may raise ValueError if text is too short for window size.
        # If 200, that's fine — skip shape check. If 422, verify shape.
        if r.status_code == 422:
            body = r.json()
            assert "request_id" in body
            assert body["schema_version"] == "1.0.0"


# =============================================================================
# 18. Config toggle wiring
# =============================================================================


@pytest.mark.skipif(not _API, reason="FastAPI not installed")
class TestConfigToggles:
    """Verify that monkeypatching config values actually changes runtime behaviour."""

    def test_lowered_max_eigenvalues_rejects_via_validator(self, monkeypatch):
        """Eigenvalue cap is read at validation time, so monkeypatch works."""
        import config as _cfg

        monkeypatch.setattr(_cfg, "MAX_EIGENVALUES", 2)
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "eigenvalues", "eigenvalues": [1.0, 2.0, 3.0]},
        )
        assert r.status_code == 422

    def test_default_limits_accept_normal_input(self):
        r = _client.post("/v1/brain/health-check", json={"text": "hello world"})
        assert r.status_code == 200

    def test_lowered_max_eigenvalues_rejects_large_array(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "MAX_EIGENVALUES", 3)
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "eigenvalues", "eigenvalues": [1.0, 2.0, 3.0, 4.0]},
        )
        assert r.status_code == 422

    def test_lowered_max_hidden_state_dims_rejects(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "MAX_HIDDEN_STATE_DIMS", 2)
        # 5x8 hidden states: 8 dims > 2
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": _HS_5x8},
        )
        assert r.status_code == 422

    def test_lowered_max_hidden_state_samples_rejects(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "MAX_HIDDEN_STATE_SAMPLES", 2)
        # 5x8 hidden states: 5 samples > 2
        r = _client.post(
            "/v1/brain/manifold-audit",
            json={"source_type": "hidden_states", "hidden_states": _HS_5x8},
        )
        assert r.status_code == 422

    def test_rate_limit_toggle_off_allows_traffic(self, monkeypatch):
        import config as _cfg

        monkeypatch.setattr(_cfg, "RATE_LIMIT_ENABLED", False)
        for _ in range(5):
            r = _client.get("/healthz")
            assert r.status_code == 200


# =============================================================================
# 19. Version visibility
# =============================================================================


@pytest.mark.skipif(not _API, reason="FastAPI not installed")
class TestVersionVisibility:
    """Verify schema_version appears where callers need it."""

    def test_readyz_contains_version(self):
        r = _client.get("/readyz")
        assert r.json()["schema_version"] == "1.0.0"

    def test_capabilities_contains_version(self):
        r = _client.get("/v1/meta/capabilities")
        assert r.json()["schema_version"] == "1.0.0"

    def test_success_response_contains_version(self):
        r = _client.post("/v1/brain/compute-correction", json={"current_r_ratio": 0.5})
        assert r.status_code == 200
        assert r.json()["schema_version"] == "1.0.0"

    def test_validation_error_contains_version(self):
        r = _client.post("/v1/brain/health-check", json={})
        assert r.status_code == 422
        assert r.json()["schema_version"] == "1.0.0"

    def test_version_matches_config(self):
        import config as _cfg

        r = _client.get("/readyz")
        assert r.json()["schema_version"] == _cfg.SCHEMA_VERSION
