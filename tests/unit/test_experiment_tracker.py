"""
tests/unit/test_experiment_tracker.py
=====================================
Unit tests for ``Experiment_Tracker`` setup, run_training, and run_inference.

Validates Requirements 8.1, 8.2, 8.3, 8.5, 9.1, 9.2, 9.3, 9.4, 10.2, 10.5.

run_training / run_inference tests require TensorFlow and are skipped when it
is absent.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

import numpy as np
import pytest
import yaml

tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed")

from network_anomaly_autoencoder.exceptions import (  # noqa: E402
    ArtifactNotFoundError,
    ConfigurationError,
)
from network_anomaly_autoencoder.experiment_tracker import Experiment_Tracker  # noqa: E402


def _sha256_bytes(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _make_yaml(tmp_path, artefact_dir: str, data_path: str, max_epochs: int = 3, patience: int = 2) -> str:
    """Write a minimal valid config and return its path."""
    config = {
        "dataset": {"path": data_path, "schema": "nsl_kdd"},
        "splits": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2, "random_seed": 42},
        "architecture": {
            "encoder_layers": [16, 8],
            "bottleneck_dim": 3,
            "activation": "relu",
            "dropout_rate": 0.1,
        },
        "training": {
            "learning_rate": 0.01,
            "batch_size": 16,
            "max_epochs": max_epochs,
            "patience": patience,
            "lr_schedule": "none",
            "random_seed": 42,
        },
        "threshold": {"percentile": 95},
        "output": {"artefact_dir": artefact_dir},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(config), encoding="utf-8")
    return str(p)


def _find_run_dir(artefact_dir: str) -> str | None:
    if not os.path.isdir(artefact_dir):
        return None
    pattern = re.compile(r"^\d{8}_\d{6}$")
    for name in os.listdir(artefact_dir):
        if pattern.match(name):
            return os.path.join(artefact_dir, name)
    return None


REQUIRED_ARTEFACTS = [
    "model_weights.keras",
    "scaler.pkl",
    "encoder.pkl",
    "threshold.json",
    "confusion_matrix.png",
    "roc_curve.png",
    "re_histogram.png",
    "metrics.json",
    "config.yaml",
]

CRITICAL_ARTEFACTS = ["model_weights.keras", "scaler.pkl", "encoder.pkl", "threshold.json"]


class TestSetup:
    """Experiment_Tracker.setup() and run-dir creation (Req 8, 9.1)."""

    def test_missing_top_level_key_raises_configuration_error(self, tmp_path):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, "/nope.csv")
        # Remove the dataset key to make the config invalid.
        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        del data["dataset"]
        with open(yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        tracker = Experiment_Tracker(yaml_path)
        with pytest.raises(ConfigurationError, match="dataset"):
            tracker.setup()
        # No run dir should be created on failure.
        assert _find_run_dir(artefact_dir) is None

    def test_create_run_dir_timestamp_pattern(self, tmp_path):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, "/nope.csv")
        tracker = Experiment_Tracker(yaml_path)
        tracker.setup()
        run_dir = _find_run_dir(artefact_dir)
        assert run_dir is not None

    def test_yaml_copied_byte_for_byte(self, tmp_path):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, "/nope.csv")
        tracker = Experiment_Tracker(yaml_path)
        tracker.setup()
        run_dir = _find_run_dir(artefact_dir)
        copied = os.path.join(run_dir, "config.yaml")
        assert os.path.isfile(copied)
        assert _sha256_bytes(yaml_path) == _sha256_bytes(copied)

    def test_log_file_created_in_run_dir(self, tmp_path):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, "/nope.csv")
        tracker = Experiment_Tracker(yaml_path)
        tracker.setup()
        run_dir = _find_run_dir(artefact_dir)
        assert os.path.isfile(os.path.join(run_dir, "experiment.log"))

    def test_git_unavailable_logged(self, tmp_path):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, "/nope.csv")
        tracker = Experiment_Tracker(yaml_path)
        tracker.setup()
        run_dir = _find_run_dir(artefact_dir)
        log_text = open(os.path.join(run_dir, "experiment.log"), encoding="utf-8").read()
        assert "git-unavailable" in log_text


class TestRunTraining:
    """run_training() produces all nine artefacts (Req 9.2)."""

    @pytest.fixture
    def trained_dir(self, tmp_path, nsl_kdd_csv_path_medium):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, nsl_kdd_csv_path_medium)
        tracker = Experiment_Tracker(yaml_path)
        tracker.run_training()
        run_dir = _find_run_dir(artefact_dir)
        return run_dir

    def test_all_nine_artefacts_present(self, trained_dir):
        assert trained_dir is not None
        for name in REQUIRED_ARTEFACTS:
            assert os.path.isfile(os.path.join(trained_dir, name)), name

    def test_metrics_json_valid(self, trained_dir):
        import json

        with open(os.path.join(trained_dir, "metrics.json"), encoding="utf-8") as fh:
            metrics = json.load(fh)
        for key in (
            "precision_normal", "recall_normal", "f1_normal",
            "precision_anomalous", "recall_anomalous", "f1_anomalous",
            "accuracy", "auc_roc", "training_time_secs",
        ):
            assert key in metrics, key
        assert isinstance(metrics["training_time_secs"], float)

    def test_error_logged_then_reraising(self, tmp_path):
        # Point dataset at a nonexistent file -> DataLoadError propagates.
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, str(tmp_path / "missing.csv"))
        tracker = Experiment_Tracker(yaml_path)
        from network_anomaly_autoencoder.exceptions import DataLoadError

        with pytest.raises(DataLoadError):
            tracker.run_training()


class TestRunInference:
    """run_inference() from saved artefacts (Req 9.3, 9.4)."""

    @pytest.fixture
    def run_dir(self, tmp_path, nsl_kdd_csv_path_medium):
        artefact_dir = str(tmp_path / "artefacts")
        yaml_path = _make_yaml(tmp_path, artefact_dir, nsl_kdd_csv_path_medium)
        tracker = Experiment_Tracker(yaml_path)
        tracker.run_training()
        return _find_run_dir(artefact_dir)

    def test_inference_predictions_shape(self, tmp_path, run_dir, nsl_kdd_csv_path_medium):
        yaml_path = _make_yaml(tmp_path, str(tmp_path / "unused"), nsl_kdd_csv_path_medium)
        tracker = Experiment_Tracker(yaml_path)
        tracker.setup()
        # Point the tracker at the previously trained run directory.
        tracker._run_dir = run_dir
        predictions, mse_scores = tracker.run_inference(nsl_kdd_csv_path_medium)
        assert predictions.shape == (400,)
        assert mse_scores.shape == (400,)

    def test_artifact_not_found_raises(self, tmp_path, run_dir, nsl_kdd_csv_path_medium):
        os.remove(os.path.join(run_dir, "model_weights.keras"))
        yaml_path = _make_yaml(tmp_path, str(tmp_path / "unused"), nsl_kdd_csv_path_medium)
        tracker = Experiment_Tracker(yaml_path)
        tracker.setup()
        tracker._run_dir = run_dir
        with pytest.raises(ArtifactNotFoundError, match="model_weights.keras"):
            tracker.run_inference(nsl_kdd_csv_path_medium)
