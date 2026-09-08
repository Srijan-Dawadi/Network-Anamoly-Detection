"""
tests/unit/test_config.py
=========================
Unit tests for ``network_anomaly_autoencoder.config.load_config``.

Validates Requirements 8.1 and 8.5:
  8.1 – load_config returns a fully populated ExperimentConfig from a valid YAML.
  8.5 – load_config raises ConfigurationError naming the offending key for any
        missing required key or out-of-range value.

Uses pytest + tmp_path fixture; YAML files are written with yaml.dump so the
test inputs are constructed programmatically and stay in sync with the schema.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from network_anomaly_autoencoder.config import (
    ArchitectureConfig,
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    SplitsConfig,
    ThresholdConfig,
    TrainingConfig,
    load_config,
)
from network_anomaly_autoencoder.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Shared valid configuration dict
# ---------------------------------------------------------------------------

_VALID_CONFIG: dict = {
    "dataset": {
        "path": "/data/KDDTrain.csv",
        "schema": "nsl_kdd",
    },
    "splits": {
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_seed": 42,
    },
    "architecture": {
        "encoder_layers": [128, 64, 32],
        "bottleneck_dim": 16,
        "activation": "relu",
        "dropout_rate": 0.2,
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
        "lr_schedule": "none",
        "random_seed": 0,
    },
    "threshold": {
        "percentile": 95,
    },
    "output": {
        "artefact_dir": "artefacts/",
    },
}


def _write_yaml(tmp_path, data: dict) -> str:
    """Dump *data* to a temp YAML file and return the path string."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# 1. Happy-path round-trip
# ---------------------------------------------------------------------------


class TestLoadConfigRoundTrip:
    """load_config returns a correct ExperimentConfig for a fully valid YAML."""

    def test_returns_experiment_config_instance(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg, ExperimentConfig)

    def test_dataset_fields(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg.dataset, DatasetConfig)
        assert cfg.dataset.path == "/data/KDDTrain.csv"
        assert cfg.dataset.schema == "nsl_kdd"

    def test_splits_fields(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg.splits, SplitsConfig)
        assert cfg.splits.train_ratio == pytest.approx(0.7)
        assert cfg.splits.val_ratio == pytest.approx(0.15)
        assert cfg.splits.test_ratio == pytest.approx(0.15)
        assert cfg.splits.random_seed == 42

    def test_architecture_fields(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg.architecture, ArchitectureConfig)
        assert cfg.architecture.encoder_layers == [128, 64, 32]
        assert cfg.architecture.bottleneck_dim == 16
        assert cfg.architecture.activation == "relu"
        assert cfg.architecture.dropout_rate == pytest.approx(0.2)

    def test_training_fields(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg.training, TrainingConfig)
        assert cfg.training.learning_rate == pytest.approx(0.001)
        assert cfg.training.batch_size == 64
        assert cfg.training.max_epochs == 100
        assert cfg.training.patience == 10
        assert cfg.training.lr_schedule == "none"
        assert cfg.training.random_seed == 0

    def test_threshold_fields(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg.threshold, ThresholdConfig)
        assert cfg.threshold.percentile == 95

    def test_output_fields(self, tmp_path):
        cfg = load_config(_write_yaml(tmp_path, _VALID_CONFIG))
        assert isinstance(cfg.output, OutputConfig)
        assert cfg.output.artefact_dir == "artefacts/"


# ---------------------------------------------------------------------------
# 2. Missing top-level keys
# ---------------------------------------------------------------------------


class TestMissingTopLevelKeys:
    """Each missing required top-level key raises ConfigurationError naming it."""

    @pytest.mark.parametrize("missing_key", [
        "dataset",
        "splits",
        "architecture",
        "training",
        "threshold",
        "output",
    ])
    def test_missing_key_raises_configuration_error(self, tmp_path, missing_key):
        data = copy.deepcopy(_VALID_CONFIG)
        del data[missing_key]
        with pytest.raises(ConfigurationError, match=missing_key):
            load_config(_write_yaml(tmp_path, data))


# ---------------------------------------------------------------------------
# 3. Out-of-range values
# ---------------------------------------------------------------------------


class TestOutOfRangeValues:
    """Out-of-range field values raise ConfigurationError naming the key."""

    # --- training.learning_rate ------------------------------------------

    def test_negative_learning_rate(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["training"]["learning_rate"] = -0.001
        with pytest.raises(ConfigurationError, match="learning_rate"):
            load_config(_write_yaml(tmp_path, data))

    def test_zero_learning_rate(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["training"]["learning_rate"] = 0.0
        with pytest.raises(ConfigurationError, match="learning_rate"):
            load_config(_write_yaml(tmp_path, data))

    # --- threshold.percentile --------------------------------------------

    def test_percentile_above_99(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["threshold"]["percentile"] = 100
        with pytest.raises(ConfigurationError, match="percentile"):
            load_config(_write_yaml(tmp_path, data))

    def test_percentile_below_1(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["threshold"]["percentile"] = 0
        with pytest.raises(ConfigurationError, match="percentile"):
            load_config(_write_yaml(tmp_path, data))

    # --- architecture.dropout_rate ---------------------------------------

    def test_dropout_rate_equal_to_one(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["architecture"]["dropout_rate"] = 1.0
        with pytest.raises(ConfigurationError, match="dropout_rate"):
            load_config(_write_yaml(tmp_path, data))

    def test_dropout_rate_above_one(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["architecture"]["dropout_rate"] = 1.5
        with pytest.raises(ConfigurationError, match="dropout_rate"):
            load_config(_write_yaml(tmp_path, data))

    # --- architecture.bottleneck_dim >= smallest encoder layer -----------

    def test_bottleneck_dim_equal_to_smallest_encoder_layer(self, tmp_path):
        # encoder_layers = [128, 64, 32], smallest = 32 → bottleneck 32 is invalid
        data = copy.deepcopy(_VALID_CONFIG)
        data["architecture"]["bottleneck_dim"] = 32
        with pytest.raises(ConfigurationError, match="bottleneck_dim"):
            load_config(_write_yaml(tmp_path, data))

    def test_bottleneck_dim_greater_than_smallest_encoder_layer(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["architecture"]["bottleneck_dim"] = 64
        with pytest.raises(ConfigurationError, match="bottleneck_dim"):
            load_config(_write_yaml(tmp_path, data))

    # --- training.batch_size ---------------------------------------------

    def test_batch_size_zero(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["training"]["batch_size"] = 0
        with pytest.raises(ConfigurationError, match="batch_size"):
            load_config(_write_yaml(tmp_path, data))

    # --- training.max_epochs ---------------------------------------------

    def test_max_epochs_zero(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["training"]["max_epochs"] = 0
        with pytest.raises(ConfigurationError, match="max_epochs"):
            load_config(_write_yaml(tmp_path, data))

    # --- training.patience -----------------------------------------------

    def test_patience_zero(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        data["training"]["patience"] = 0
        with pytest.raises(ConfigurationError, match="patience"):
            load_config(_write_yaml(tmp_path, data))

    # --- splits: ratios that don't sum to 1.0 ----------------------------

    def test_split_ratios_do_not_sum_to_one(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        # 0.6 + 0.2 + 0.1 = 0.9, not 1.0
        data["splits"]["train_ratio"] = 0.6
        data["splits"]["val_ratio"] = 0.2
        data["splits"]["test_ratio"] = 0.1
        with pytest.raises(ConfigurationError, match="splits"):
            load_config(_write_yaml(tmp_path, data))

    # --- splits: a ratio of exactly 0.0 ----------------------------------

    def test_split_ratio_of_zero(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        # train_ratio = 0.0 is outside (0.0, 1.0) exclusive range
        data["splits"]["train_ratio"] = 0.0
        data["splits"]["val_ratio"] = 0.5
        data["splits"]["test_ratio"] = 0.5
        with pytest.raises(ConfigurationError, match="train_ratio"):
            load_config(_write_yaml(tmp_path, data))

    # --- splits: a ratio of exactly 1.0 ----------------------------------

    def test_split_ratio_of_one(self, tmp_path):
        data = copy.deepcopy(_VALID_CONFIG)
        # val_ratio = 1.0 is outside (0.0, 1.0) exclusive range
        data["splits"]["train_ratio"] = 0.0
        data["splits"]["val_ratio"] = 1.0
        data["splits"]["test_ratio"] = 0.0
        with pytest.raises(ConfigurationError, match="train_ratio"):
            load_config(_write_yaml(tmp_path, data))
