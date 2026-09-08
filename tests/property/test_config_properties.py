"""
tests/property/test_config_properties.py
=========================================
Hypothesis-powered property tests for ``load_config``.

      Feature: network-anomaly-autoencoder
      Property 18: Missing YAML Key Raises ConfigurationError Naming the Key

Validates Requirements 8.1, 8.5.
"""

from __future__ import annotations

import uuid

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from network_anomaly_autoencoder.config import load_config
from network_anomaly_autoencoder.exceptions import ConfigurationError

TOP_LEVEL_KEYS = ["dataset", "splits", "architecture", "training", "threshold", "output"]


@pytest.fixture(scope="session")
def _tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("cfg_props")


def _valid_config() -> dict:
    return {
        "dataset": {"path": "/data/x.csv", "schema": "nsl_kdd"},
        "splits": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_seed": 42},
        "architecture": {
            "encoder_layers": [64, 32], "bottleneck_dim": 8,
            "activation": "relu", "dropout_rate": 0.2,
        },
        "training": {
            "learning_rate": 0.001, "batch_size": 32, "max_epochs": 10,
            "patience": 3, "lr_schedule": "none", "random_seed": 0,
        },
        "threshold": {"percentile": 95},
        "output": {"artefact_dir": "runs"},
    }


@given(st.sampled_from(TOP_LEVEL_KEYS))
@settings(max_examples=100)
def test_missing_top_level_key_raises_configuration_error_naming_key(
    _tmp, missing_key: str
):
    # Feature: network-anomaly-autoencoder, Property 18: Missing YAML Key
    config = _valid_config()
    del config[missing_key]
    path = _tmp / f"{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")

    try:
        load_config(str(path))
        assert False, f"Expected ConfigurationError for missing key '{missing_key}'"
    except ConfigurationError as exc:
        assert missing_key in str(exc), (
            f"Error message should name the missing key '{missing_key}', got: {exc}"
        )
