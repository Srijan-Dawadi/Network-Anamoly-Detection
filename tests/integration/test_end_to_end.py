"""
tests/integration/test_end_to_end.py
=====================================
End-to-end integration tests for the network-anomaly-autoencoder.

      Feature: network-anomaly-autoencoder
      16.1 End-to-end training integration test
      16.2 Inference-from-artefact integration test
      16.3 CPU inference timing integration test
      16.4 Early-stopping integration test

Validates Requirements 4.3, 6.4, 7.6, 9.2, 9.3.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

import numpy as np
import pytest
import yaml

from network_anomaly_autoencoder.config import load_config
from network_anomaly_autoencoder.data_pipeline import Data_Pipeline
from network_anomaly_autoencoder.experiment_tracker import Experiment_Tracker
from network_anomaly_autoencoder.model import Autoencoder_Model

from tests.conftest import _make_nsl_kdd_df

logger = logging.getLogger("integration")

_ALL_NINE_ARTEFACTS = [
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

_METRIC_KEYS = (
    "precision_normal",
    "recall_normal",
    "f1_normal",
    "precision_anomalous",
    "recall_anomalous",
    "f1_anomalous",
    "accuracy",
    "auc_roc",
)


@pytest.fixture(scope="module")
def _tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("integration")


def _write_csv(path: str, n_normal: int, n_attack: int) -> None:
    df = _make_nsl_kdd_df(n_normal=n_normal, n_attack=n_attack)
    df.to_csv(path, index=False)


def _small_config(artefact_dir: str, data_path: str, max_epochs: int = 5, patience: int = 3):
    return {
        "dataset": {"path": data_path, "schema": "nsl_kdd"},
        "splits": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2, "random_seed": 42},
        "architecture": {
            "encoder_layers": [32, 16], "bottleneck_dim": 8,
            "activation": "relu", "dropout_rate": 0.2,
        },
        "training": {
            "learning_rate": 0.001, "batch_size": 32, "max_epochs": max_epochs,
            "patience": patience, "lr_schedule": "none", "random_seed": 0,
        },
        "threshold": {"percentile": 95},
        "output": {"artefact_dir": artefact_dir},
    }


def _write_config(dirpath: str, filename: str, config: dict, data_path: str) -> str:
    config["dataset"]["path"] = data_path
    path = os.path.join(dirpath, filename)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh)
    return path


def _feature_dim(data_path: str) -> int:
    """Return the post-preprocessing feature dimensionality for a dataset."""
    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(
        cfg,
        dataset=dataclasses.replace(cfg.dataset, path=data_path),
    )
    pipeline = Data_Pipeline(cfg, logger)
    df = pipeline.load_and_validate()
    Xtr, *_ = pipeline.fit_transform(df)
    return Xtr.shape[1]


@pytest.fixture(scope="module")
def trained_run(_tmp):
    """Run the full training pipeline once and expose its artifacts + helpers."""
    base = str(_tmp / uuid.uuid4().hex)
    os.makedirs(base, exist_ok=True)

    train_csv = os.path.join(base, "train.csv")
    infer_csv = os.path.join(base, "infer.csv")
    _write_csv(train_csv, 300, 100)
    _write_csv(infer_csv, 20, 10)

    config = _small_config(base, train_csv)
    config_path = _write_config(base, "config.yaml", config, train_csv)

    tracker = Experiment_Tracker(config_path)
    config_obj = tracker.setup()
    run_dir = tracker._run_dir
    assert run_dir is not None
    tracker.run_training()

    feature_dim = _feature_dim(train_csv)

    class Result:
        pass

    result = Result()
    result.base = base
    result.run_dir = run_dir
    result.config_path = config_path
    result.config_obj = config_obj
    result.train_csv = train_csv
    result.infer_csv = infer_csv
    result.feature_dim = feature_dim
    return result


# ---------------------------------------------------------------------------
# 16.1 End-to-end training
# ---------------------------------------------------------------------------


def test_16_1_e2e_training_produces_all_artefacts(trained_run):
    # Feature: network-anomaly-autoencoder, 16.1
    run_dir = trained_run.run_dir
    for name in _ALL_NINE_ARTEFACTS:
        assert os.path.isfile(os.path.join(run_dir, name)), f"Missing artefact: {name}"

    with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    for key in _METRIC_KEYS:
        assert key in metrics, f"metrics.json missing key: {key}"
        assert 0.0 <= float(metrics[key]) <= 1.0, (
            f"Metric '{key}' out of range: {metrics[key]}"
        )


# ---------------------------------------------------------------------------
# 16.2 Inference from artifacts
# ---------------------------------------------------------------------------


def test_16_2_inference_from_artefacts(trained_run):
    # Feature: network-anomaly-autoencoder, 16.2
    tracker = Experiment_Tracker(trained_run.config_path)
    tracker.setup()
    # Point the tracker at the training run's artefact directory so it can
    # load the model weights / scaler / encoder / threshold produced in 16.1.
    tracker._run_dir = trained_run.run_dir

    predictions, mse_scores = tracker.run_inference(trained_run.infer_csv)
    assert predictions.shape[0] == 30  # 20 normal + 10 attack input rows
    assert mse_scores.shape[0] == 30
    assert set(np.unique(predictions)) <= {0, 1}
    assert np.all(mse_scores >= 0)


# ---------------------------------------------------------------------------
# 16.3 CPU inference timing (< 5s for 10k rows)
# ---------------------------------------------------------------------------


def test_16_3_cpu_inference_timing(trained_run):
    # Feature: network-anomaly-autoencoder, 16.3
    config = trained_run.config_obj
    arch = config.architecture
    train = config.training
    model = Autoencoder_Model(arch, train, logger)
    model.build(trained_run.feature_dim)
    model.load_weights(os.path.join(trained_run.run_dir, "model_weights.keras"))

    rng = np.random.default_rng(7)
    X = rng.normal(size=(10_000, trained_run.feature_dim)).astype("float32")

    # Warm-up pass, then measure.
    model.predict(X, 0.5)
    start = time.perf_counter()
    model.predict(X, 0.5)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"CPU inference took {elapsed:.3f}s (must be < 5s)"


# ---------------------------------------------------------------------------
# 16.4 Early stopping
# ---------------------------------------------------------------------------


def test_16_4_early_stopping(_tmp, monkeypatch):
    # Feature: network-anomaly-autoencoder, 16.4
    base = str(_tmp / uuid.uuid4().hex)
    os.makedirs(base, exist_ok=True)
    data_path = os.path.join(base, "overfit.csv")
    _write_csv(data_path, 500, 0)

    # max_epochs=200, patience=3, overfit-prone capacity (no dropout, wide net).
    config_path = _write_config(
        base, "es_config.yaml",
        _small_config(base, data_path, max_epochs=200, patience=3),
        data_path,
    )
    cfg = load_config(config_path)
    import dataclasses

    cfg = dataclasses.replace(
        cfg,
        architecture=dataclasses.replace(
            cfg.architecture, encoder_layers=[64, 32], bottleneck_dim=16, dropout_rate=0.0
        ),
        training=dataclasses.replace(cfg.training, batch_size=16),
    )

    pipeline = Data_Pipeline(cfg, logger)
    df = pipeline.load_and_validate()
    Xtr, Xval, *_ = pipeline.fit_transform(df)

    model = Autoencoder_Model(cfg.architecture, cfg.training, logger)
    model.build(Xtr.shape[1])

    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")
    history = model.fit(Xtr, Xval, base)
    assert len(history.epoch) < 200, (
        f"Early stopping did not trigger: trained for {len(history.epoch)} epochs"
    )
    assert len(history.epoch) >= 1
