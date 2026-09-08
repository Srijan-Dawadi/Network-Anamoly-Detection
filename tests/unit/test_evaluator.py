"""
tests/unit/test_evaluator.py
============================
Unit tests for ``Evaluator``.

Validates Requirements 7.1, 7.3, 7.4, 7.5, 7.6, 7.7.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest

from network_anomaly_autoencoder.evaluator import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator(logger=logging.getLogger("test"))


class TestEvaluateErrors:
    def test_length_mismatch_raises_value_error(self, evaluator):
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])
        mse = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate(y_true, y_pred, mse, 0.5, ".")


class TestEvaluateOutputs:
    def test_png_artefacts_created(self, evaluator, tmp_path):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        mse = np.array([0.1, 0.2, 0.3, 1.0, 2.0])
        evaluator.evaluate(y_true, y_pred, mse, 0.5, str(tmp_path))
        for name in ("confusion_matrix.png", "roc_curve.png", "re_histogram.png"):
            assert os.path.isfile(os.path.join(str(tmp_path), name))

    def test_metrics_json_all_keys_in_range(self, evaluator, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        mse = np.array([0.1, 0.5, 1.0, 2.0])
        evaluator.evaluate(y_true, y_pred, mse, 0.5, str(tmp_path))

        with open(os.path.join(str(tmp_path), "metrics.json"), "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        required = {
            "precision_normal",
            "recall_normal",
            "f1_normal",
            "precision_anomalous",
            "recall_anomalous",
            "f1_anomalous",
            "accuracy",
            "auc_roc",
        }
        assert required.issubset(set(metrics.keys()))
        for value in metrics.values():
            assert 0.0 <= value <= 1.0

    def test_perfectly_separable_auc_is_one(self, evaluator, tmp_path):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        mse = np.array([0.1, 0.2, 0.3, 5.0, 6.0, 7.0])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        metrics = evaluator.evaluate(y_true, y_pred, mse, 0.5, str(tmp_path))
        assert metrics["auc_roc"] == pytest.approx(1.0)

    def test_evaluate_returns_metrics_dict(self, evaluator, tmp_path):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        mse = np.array([0.1, 0.2, 1.0, 2.0])
        metrics = evaluator.evaluate(y_true, y_pred, mse, 0.5, str(tmp_path))
        assert metrics["accuracy"] == pytest.approx(1.0)


class TestAucWarning:
    def test_warning_logged_when_auc_below_08(self, evaluator, tmp_path, caplog):
        caplog.set_level("WARNING")
        y_true = np.array([0, 0, 0, 1, 1, 1])
        mse = np.array([9.0, 8.0, 7.0, 0.1, 0.2, 0.3])  # inverse ranking -> AUC ~0
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        evaluator.evaluate(y_true, y_pred, mse, 0.5, str(tmp_path))
        assert any("underperforming" in r.message or "AUC-ROC" in r.message for r in caplog.records)
