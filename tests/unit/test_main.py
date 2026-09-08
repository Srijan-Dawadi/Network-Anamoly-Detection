"""
tests/unit/test_main.py
=======================
Unit tests for the ``main.py`` CLI entry point.

Validates Requirements 8.1 (CLI routing and error handling).
"""

from __future__ import annotations

import sys

import pytest

import main as main_module
from network_anomaly_autoencoder.exceptions import ConfigurationError


class _FakeTracker:
    """Replaces Experiment_Tracker for routing tests."""

    def __init__(self, config):
        self.calls = []

    def run_training(self):
        self.calls.append("train")

    def run_inference(self, data):
        self.calls.append(("infer", data))
        import numpy as np

        return np.array([0, 0, 1]), np.array([0.1, 0.2, 0.9])


@pytest.fixture
def fake_tracker(monkeypatch):
    tracker = _FakeTracker(None)
    monkeypatch.setattr(main_module, "Experiment_Tracker", lambda cfg: tracker)
    return tracker


class TestRouting:
    def test_train_subcommand_routes_to_run_training(self, fake_tracker, capsys):
        rc = main_module.main(["train", "--config", "configs/nsl_kdd_default.yaml"])
        assert rc == 0
        assert fake_tracker.calls == ["train"]
        assert "Training completed" in capsys.readouterr().out

    def test_infer_subcommand_routes_to_run_inference(self, fake_tracker):
        rc = main_module.main(
            ["infer", "--config", "configs/nsl_kdd_default.yaml", "--data", "data/x.csv"]
        )
        assert rc == 0
        assert fake_tracker.calls == [("infer", "data/x.csv")]


class TestErrorHandling:
    def test_network_anomaly_error_returns_1_and_stderr(self, monkeypatch, capsys):
        def boom(self):
            raise ConfigurationError("bad key: dataset")
        monkeypatch.setattr(main_module.Experiment_Tracker, "run_training", boom)
        rc = main_module.main(["train", "--config", "configs/nsl_kdd_default.yaml"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "bad key: dataset" in err

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main_module.main([])
