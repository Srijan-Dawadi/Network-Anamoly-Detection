"""
tests/property/test_experiment_tracker_properties.py
=====================================================
Hypothesis-powered property tests covering the Experiment_Tracker.

      Feature: network-anomaly-autoencoder
      Property 15: ArtifactNotFoundError Lists All Missing Files
      Property 16: Artefact Completeness After Training
      Property 17: Config File Byte-for-Byte Copy

Validates Requirements 8.2, 9.2, 9.4.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from network_anomaly_autoencoder.exceptions import ArtifactNotFoundError
from network_anomaly_autoencoder.experiment_tracker import Experiment_Tracker

from tests.conftest import _make_nsl_kdd_df

logger = logging.getLogger("property")

_CRITICAL = ["model_weights.keras", "scaler.pkl", "encoder.pkl", "threshold.json"]

_REQUIRED_ARTEFACTS = [
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


@pytest.fixture(scope="session")
def _tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("expt_props")


def _newdir(_tmp) -> str:
    d = _tmp / uuid.uuid4().hex
    d.mkdir(parents=True)
    return str(d)


def _base_config(artefact_dir: str) -> dict:
    return {
        "dataset": {"path": "data/KDDTrain+.csv", "schema": "nsl_kdd"},
        "splits": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2, "random_seed": 42},
        "architecture": {
            "encoder_layers": [8, 4], "bottleneck_dim": 2,
            "activation": "relu", "dropout_rate": 0.0,
        },
        "training": {
            "learning_rate": 0.001, "batch_size": 8, "max_epochs": 2,
            "patience": 2, "lr_schedule": "none", "random_seed": 0,
        },
        "threshold": {"percentile": 95},
        "output": {"artefact_dir": artefact_dir},
    }


def _write_config(dirpath: str, filename: str, config: dict) -> str:
    path = os.path.join(dirpath, filename)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh)
    return path


def _write_csv(path: str, n_normal: int = 200, n_attack: int = 50) -> None:
    df = _make_nsl_kdd_df(n_normal=n_normal, n_attack=n_attack)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Property 15: ArtifactNotFoundError lists exactly the missing files
# ---------------------------------------------------------------------------

_nonempty_subsets = st.lists(
    st.sampled_from(_CRITICAL), min_size=1, max_size=len(_CRITICAL), unique=True
)


@given(to_delete=_nonempty_subsets)
@settings(max_examples=20, deadline=None)
def test_property15_missing_artefacts_listed(_tmp, to_delete):
    # Feature: network-anomaly-autoencoder, Property 15
    base = _newdir(_tmp)
    csv_path = os.path.join(base, "data.csv")
    _write_csv(csv_path)
    config_path = _write_config(base, "cfg.yaml", _base_config(base))

    tracker = Experiment_Tracker(config_path)
    tracker.setup()
    run_dir = tracker._run_dir
    assert run_dir is not None

    # Populate all four critical artefacts, then delete the chosen subset.
    for name in _CRITICAL:
        with open(os.path.join(run_dir, name), "w") as fh:
            fh.write("placeholder")
    for name in to_delete:
        os.remove(os.path.join(run_dir, name))

    missing_expected = sorted(to_delete)
    try:
        tracker.run_inference(csv_path)
        assert False, "Expected ArtifactNotFoundError for missing artefacts"
    except ArtifactNotFoundError as exc:
        msg = str(exc)
        for name in missing_expected:
            assert name in msg, f"ArtifactNotFoundError should list '{name}': {msg}"
        # The message must list all and only the absent files.
        names_in_msg = [name for name in _CRITICAL if name in msg]
        assert sorted(names_in_msg) == missing_expected


# ---------------------------------------------------------------------------
# Property 16: Artefact completeness after training
# ---------------------------------------------------------------------------


def test_property16_artefact_completeness_after_training(_tmp):
    # Feature: network-anomaly-autoencoder, Property 16
    base = _newdir(_tmp)
    csv_path = os.path.join(base, "data.csv")
    _write_csv(csv_path, n_normal=400, n_attack=100)
    config = _base_config(base)
    config["dataset"]["path"] = csv_path
    # Name the source config `config.yaml` so the byte-for-byte copy lands at
    # the expected `config.yaml` artefact path (Requirement 9.2 / Property 16).
    config_path = _write_config(base, "config.yaml", config)

    tracker = Experiment_Tracker(config_path)
    tracker.setup()
    run_dir = tracker._run_dir
    assert run_dir is not None
    tracker.run_training()

    for name in _REQUIRED_ARTEFACTS:
        assert os.path.isfile(os.path.join(run_dir, name)), f"Missing artefact: {name}"


# ---------------------------------------------------------------------------
# Property 17: Config file byte-for-byte copy
# ---------------------------------------------------------------------------

def test_property17_config_byte_for_byte_copy(_tmp):
    # Feature: network-anomaly-autoencoder, Property 17
    base = _newdir(_tmp)
    filename = f"custom_{uuid.uuid4().hex}.yaml"
    config = _base_config(base)
    config_path = _write_config(base, filename, config)

    with open(config_path, "rb") as fh:
        original_bytes = fh.read()

    tracker = Experiment_Tracker(config_path)
    tracker.setup()
    run_dir = tracker._run_dir
    assert run_dir is not None

    copy_path = os.path.join(run_dir, filename)
    assert os.path.isfile(copy_path)
    with open(copy_path, "rb") as fh:
        copied_bytes = fh.read()

    assert copied_bytes == original_bytes
    assert hashlib.sha256(copied_bytes).digest() == hashlib.sha256(original_bytes).digest()
    assert os.path.basename(copy_path) == filename
