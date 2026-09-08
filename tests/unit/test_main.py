"""
tests/unit/test_main.py
=======================
Unit tests for the ``main.py`` CLI entry point.

Validates Requirements 8.1 (CLI routing and error handling).
"""

from __future__ import annotations

import json
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

    def run_inference(self, data, run_dir=None):
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

    def test_infer_subcommand_routes_to_run_inference(self, fake_tracker, monkeypatch):
        monkeypatch.setattr(main_module, "_latest_run_dir", lambda _: "runs/00000000_000000")
        rc = main_module.main(
            ["infer", "--config", "configs/nsl_kdd_default.yaml", "--data", "data/x.csv"]
        )
        assert rc == 0
        assert fake_tracker.calls == [("infer", "data/x.csv")]

    def test_infer_with_explicit_run_dir_routes(self, fake_tracker):
        rc = main_module.main(
            [
                "infer",
                "--config", "configs/nsl_kdd_default.yaml",
                "--data", "data/x.csv",
                "--run-dir", "runs/custom_run",
            ]
        )
        assert rc == 0
        assert fake_tracker.calls == [("infer", "data/x.csv")]

    def test_infer_no_runs_returns_1(self, fake_tracker, monkeypatch, capsys):
        def no_runs(_):
            raise FileNotFoundError("Runs not found")
        monkeypatch.setattr(main_module, "_latest_run_dir", no_runs)
        rc = main_module.main(
            ["infer", "--config", "configs/nsl_kdd_default.yaml", "--data", "data/x.csv"]
        )
        assert rc == 1
        assert "Runs not found" in capsys.readouterr().err


class TestReport:
    """report subcommand routes to the printer and handles missing runs (Req 8.1)."""

    def test_report_defaults_to_latest_run(self, fake_tracker, monkeypatch):
        calls = {}

        def fake_print(run_dir, open_plots=False):
            calls["run_dir"] = run_dir
            calls["open_plots"] = open_plots

        monkeypatch.setattr(main_module, "_latest_run_dir", lambda _: "runs/20260101_000000")
        monkeypatch.setattr(main_module, "_print_experiment_report", fake_print)
        rc = main_module.main(
            ["report", "--config", "configs/nsl_kdd_default.yaml"]
        )
        assert rc == 0
        assert calls == {"run_dir": "runs/20260101_000000", "open_plots": False}

    def test_report_explicit_run_dir_routes(self, fake_tracker, monkeypatch):
        calls = {}

        def fake_print(run_dir, open_plots=False):
            calls["run_dir"] = run_dir
            calls["open_plots"] = open_plots

        def should_not_be_called(_):
            raise AssertionError("_latest_run_dir should be skipped with --run-dir")

        monkeypatch.setattr(main_module, "_latest_run_dir", should_not_be_called)
        monkeypatch.setattr(main_module, "_print_experiment_report", fake_print)
        rc = main_module.main(
            [
                "report",
                "--config", "configs/nsl_kdd_default.yaml",
                "--run-dir", "runs/custom",
                "--open",
            ]
        )
        assert rc == 0
        assert calls == {"run_dir": "runs/custom", "open_plots": True}

    def test_report_no_runs_returns_1(self, fake_tracker, monkeypatch, capsys):
        def no_runs(_):
            raise FileNotFoundError("Runs not found")

        monkeypatch.setattr(main_module, "_latest_run_dir", no_runs)
        rc = main_module.main(
            ["report", "--config", "configs/nsl_kdd_default.yaml"]
        )
        assert rc == 1
        assert "Runs not found" in capsys.readouterr().err

    def test_report_prints_metrics_from_json(self, tmp_path, capsys):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "precision_normal": 0.91,
                    "recall_normal": 0.94,
                    "f1_normal": 0.92,
                    "precision_anomalous": 0.93,
                    "recall_anomalous": 0.90,
                    "f1_anomalous": 0.91,
                    "accuracy": 0.92,
                    "auc_roc": 0.97,
                    "training_time_secs": 18.3,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "threshold.json").write_text(
            json.dumps(
                {
                    "threshold": 0.35,
                    "percentile": 95,
                    "mse_mean": 0.21,
                    "mse_std": 5.5,
                }
            ),
            encoding="utf-8",
        )

        from main import _print_experiment_report

        _print_experiment_report(str(run_dir))
        out = capsys.readouterr().out
        assert "Accuracy" in out and "92.00%" in out
        assert "AUC-ROC" in out and "0.9700" in out
        assert "0.350000" in out
        assert "Precision" in out and "Recall" in out

    def test_report_missing_metrics_raises_file_not_found(self, tmp_path):
        from main import _print_experiment_report

        with pytest.raises(FileNotFoundError, match="metrics.json"):
            _print_experiment_report(str(tmp_path))


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
