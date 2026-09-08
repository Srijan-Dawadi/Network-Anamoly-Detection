"""
tests/unit/test_threshold.py
============================
Unit tests for ``Threshold_Estimator``.

Validates Requirements 5.4, 5.5.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from network_anomaly_autoencoder.config import ThresholdConfig
from network_anomaly_autoencoder.exceptions import (
    ArtifactLoadError,
    InferenceError,
)
from network_anomaly_autoencoder.threshold import Threshold_Estimator


def _make_estimator(percentile: int = 95, logger: logging.Logger | None = None) -> Threshold_Estimator:
    return Threshold_Estimator(
        config=ThresholdConfig(percentile=percentile),
        logger=logger or logging.getLogger("test"),
    )


def _write_threshold_json(tmp_path, payload):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestFit:
    def test_fit_computes_percentile(self):
        est = _make_estimator(percentile=95)
        mse = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 2.0, 3.0])
        threshold = est.fit(mse)
        assert abs(threshold - np.percentile(mse, 95)) < 1e-9

    def test_fit_stores_statistics(self):
        est = _make_estimator(percentile=90)
        mse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        est.fit(mse)
        assert abs(est._mse_mean - np.mean(mse)) < 1e-9
        assert abs(est._mse_std - np.std(mse)) < 1e-9

    def test_percentile_boundary_1(self):
        est = _make_estimator(percentile=1)
        mse = np.arange(1, 101, dtype=float)
        assert abs(est.fit(mse) - np.percentile(mse, 1)) < 1e-9

    def test_percentile_boundary_99(self):
        est = _make_estimator(percentile=99)
        mse = np.arange(1, 101, dtype=float)
        assert abs(est.fit(mse) - np.percentile(mse, 99)) < 1e-9

    def test_threshold_property_raises_before_fit(self):
        est = _make_estimator()
        with pytest.raises(InferenceError, match="not yet computed or loaded"):
            _ = est.threshold

    def test_threshold_property_after_fit(self):
        est = _make_estimator()
        est.fit(np.array([1.0, 2.0, 3.0, 4.0]))
        assert est.threshold >= 0


class TestSaveLoad:
    def test_save_writes_all_four_fields(self, tmp_path):
        est = _make_estimator(percentile=95)
        est.fit(np.array([0.5, 1.0, 1.5, 2.0, 5.0]))
        est.save(str(tmp_path))

        data = json.loads((tmp_path / "threshold.json").read_text(encoding="utf-8"))
        assert set(data.keys()) == {"threshold", "percentile", "mse_mean", "mse_std"}

    def test_load_round_trip(self, tmp_path):
        est = _make_estimator(percentile=90)
        mse = np.array([0.1, 0.2, 0.3, 1.0, 2.0])
        est.fit(mse)
        est.save(str(tmp_path))

        loaded = _make_estimator(percentile=90)
        val = loaded.load(str(tmp_path))
        assert abs(val - est.threshold) < 1e-9

    def test_load_missing_field_raises(self, tmp_path):
        _write_threshold_json(tmp_path, {"threshold": 0.5, "percentile": 95})
        est = _make_estimator()
        with pytest.raises(ArtifactLoadError, match="mse_mean"):
            est.load(str(tmp_path))

    def test_load_invalid_json_raises(self, tmp_path):
        (tmp_path / "threshold.json").write_text("{not json", encoding="utf-8")
        est = _make_estimator()
        with pytest.raises(ArtifactLoadError, match="invalid JSON"):
            est.load(str(tmp_path))

    def test_load_unparseable_field_raises(self, tmp_path):
        _write_threshold_json(
            tmp_path,
            {"threshold": "abc", "percentile": 95, "mse_mean": 0.1, "mse_std": 0.2},
        )
        est = _make_estimator()
        with pytest.raises(ArtifactLoadError, match="unparseable"):
            est.load(str(tmp_path))

    def test_load_missing_file_raises(self, tmp_path):
        est = _make_estimator()
        with pytest.raises(ArtifactLoadError, match="not found"):
            est.load(str(tmp_path))
