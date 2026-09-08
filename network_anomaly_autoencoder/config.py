"""
config.py
=========
Configuration dataclasses and YAML loader for the network-anomaly-autoencoder
experiment pipeline.

Usage
-----
    from network_anomaly_autoencoder.config import load_config

    cfg = load_config("configs/nsl_kdd_default.yaml")
    print(cfg.training.learning_rate)

All validation errors raise ``ConfigurationError`` with a message that names
the offending key or parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import yaml

from network_anomaly_autoencoder.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Allowed values for enumerated string fields
# ---------------------------------------------------------------------------
_VALID_SCHEMAS: frozenset[str] = frozenset({"nsl_kdd", "cicids2017", "unsw_nb15"})
_VALID_LR_SCHEDULES: frozenset[str] = frozenset({"none", "step_decay", "cosine_annealing"})

# Required top-level YAML keys (order matches the design document)
_REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "dataset",
    "splits",
    "architecture",
    "training",
    "threshold",
    "output",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for the raw dataset source."""

    path: str
    """Absolute or relative path to the CSV file."""

    schema: str
    """Dataset schema name. One of: ``"nsl_kdd"``, ``"cicids2017"``, ``"unsw_nb15"``."""


@dataclass
class SplitsConfig:
    """Train / validation / test split configuration."""

    train_ratio: float
    """Fraction of normal traffic used for training. Exclusive range (0.0, 1.0)."""

    val_ratio: float
    """Fraction of normal traffic used for validation. Exclusive range (0.0, 1.0)."""

    test_ratio: float
    """Fraction used for the test set (normal + attack). Exclusive range (0.0, 1.0)."""

    random_seed: int
    """Random seed for reproducible splits."""


@dataclass
class ArchitectureConfig:
    """Autoencoder architecture hyperparameters."""

    encoder_layers: List[int]
    """Hidden layer sizes on the encoder side, e.g. ``[128, 64, 32]``.
    Each value must be >= 1."""

    bottleneck_dim: int
    """Dimensionality of the bottleneck (latent) layer. Must be >= 1 and
    strictly less than the smallest encoder layer size."""

    activation: str
    """Activation function name, e.g. ``"relu"`` or ``"tanh"``."""

    dropout_rate: float
    """Dropout probability. Valid range [0.0, 1.0)."""


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    learning_rate: float
    """Optimiser learning rate. Must be > 0.0."""

    batch_size: int
    """Mini-batch size. Must be >= 1."""

    max_epochs: int
    """Maximum number of training epochs. Must be >= 1."""

    patience: int
    """Early-stopping patience (number of epochs without improvement). Must be >= 1."""

    lr_schedule: str
    """Learning-rate schedule. One of: ``"none"``, ``"step_decay"``,
    ``"cosine_annealing"``."""

    random_seed: int
    """Random seed for weight initialisation and training stochasticity."""


@dataclass
class ThresholdConfig:
    """Anomaly threshold estimation configuration."""

    percentile: int
    """Percentile of training-set MSE distribution used as the anomaly threshold.
    Valid range [1, 99] inclusive. Default is 95."""


@dataclass
class OutputConfig:
    """Output and artefact storage configuration."""

    artefact_dir: str
    """Base directory under which timestamped run subdirectories are created."""


@dataclass
class ExperimentConfig:
    """Top-level configuration object aggregating all sub-configs."""

    dataset: DatasetConfig
    splits: SplitsConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    threshold: ThresholdConfig
    output: OutputConfig


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _require_key(mapping: dict, key: str, context: str = "") -> object:
    """Return ``mapping[key]`` or raise ``ConfigurationError`` naming the key."""
    if key not in mapping:
        prefix = f"{context}." if context else ""
        raise ConfigurationError(
            f"Missing required configuration key: '{prefix}{key}'"
        )
    return mapping[key]


def _validate_dataset(raw: dict) -> DatasetConfig:
    path = _require_key(raw, "path", "dataset")
    schema = _require_key(raw, "schema", "dataset")

    if not isinstance(path, str) or not path.strip():
        raise ConfigurationError(
            "Configuration key 'dataset.path' must be a non-empty string."
        )
    if schema not in _VALID_SCHEMAS:
        raise ConfigurationError(
            f"Configuration key 'dataset.schema' has invalid value '{schema}'. "
            f"Must be one of: {sorted(_VALID_SCHEMAS)}."
        )
    return DatasetConfig(path=path, schema=schema)


def _validate_splits(raw: dict) -> SplitsConfig:
    train_ratio = _require_key(raw, "train_ratio", "splits")
    val_ratio = _require_key(raw, "val_ratio", "splits")
    test_ratio = _require_key(raw, "test_ratio", "splits")
    random_seed = _require_key(raw, "random_seed", "splits")

    for name, value in [
        ("splits.train_ratio", train_ratio),
        ("splits.val_ratio", val_ratio),
        ("splits.test_ratio", test_ratio),
    ]:
        if not isinstance(value, (int, float)) or not (0.0 < float(value) < 1.0):
            raise ConfigurationError(
                f"Configuration key '{name}' must be in the exclusive range "
                f"(0.0, 1.0), got: {value}."
            )

    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-6:
        raise ConfigurationError(
            f"Configuration keys 'splits.train_ratio', 'splits.val_ratio', and "
            f"'splits.test_ratio' must sum to 1.0, but they sum to {total:.6f}."
        )

    if not isinstance(random_seed, int):
        raise ConfigurationError(
            f"Configuration key 'splits.random_seed' must be an integer, "
            f"got: {type(random_seed).__name__}."
        )

    return SplitsConfig(
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        random_seed=random_seed,
    )


def _validate_architecture(raw: dict) -> ArchitectureConfig:
    encoder_layers = _require_key(raw, "encoder_layers", "architecture")
    bottleneck_dim = _require_key(raw, "bottleneck_dim", "architecture")
    activation = _require_key(raw, "activation", "architecture")
    dropout_rate = _require_key(raw, "dropout_rate", "architecture")

    if (
        not isinstance(encoder_layers, list)
        or len(encoder_layers) == 0
        or not all(isinstance(v, int) and v >= 1 for v in encoder_layers)
    ):
        raise ConfigurationError(
            "Configuration key 'architecture.encoder_layers' must be a non-empty "
            "list of integers each >= 1."
        )

    if not isinstance(bottleneck_dim, int) or bottleneck_dim < 1:
        raise ConfigurationError(
            f"Configuration key 'architecture.bottleneck_dim' must be an integer "
            f">= 1, got: {bottleneck_dim}."
        )

    min_encoder = min(encoder_layers)
    if bottleneck_dim >= min_encoder:
        raise ConfigurationError(
            f"Configuration key 'architecture.bottleneck_dim' ({bottleneck_dim}) "
            f"must be strictly less than the smallest encoder layer size "
            f"({min_encoder})."
        )

    if not isinstance(activation, str) or not activation.strip():
        raise ConfigurationError(
            "Configuration key 'architecture.activation' must be a non-empty string."
        )

    if (
        not isinstance(dropout_rate, (int, float))
        or not (0.0 <= float(dropout_rate) < 1.0)
    ):
        raise ConfigurationError(
            f"Configuration key 'architecture.dropout_rate' must be in the range "
            f"[0.0, 1.0), got: {dropout_rate}."
        )

    return ArchitectureConfig(
        encoder_layers=list(encoder_layers),
        bottleneck_dim=bottleneck_dim,
        activation=activation,
        dropout_rate=float(dropout_rate),
    )


def _validate_training(raw: dict) -> TrainingConfig:
    learning_rate = _require_key(raw, "learning_rate", "training")
    batch_size = _require_key(raw, "batch_size", "training")
    max_epochs = _require_key(raw, "max_epochs", "training")
    patience = _require_key(raw, "patience", "training")
    lr_schedule = _require_key(raw, "lr_schedule", "training")
    random_seed = _require_key(raw, "random_seed", "training")

    if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0.0:
        raise ConfigurationError(
            f"Configuration key 'training.learning_rate' must be > 0.0, "
            f"got: {learning_rate}."
        )

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ConfigurationError(
            f"Configuration key 'training.batch_size' must be an integer >= 1, "
            f"got: {batch_size}."
        )

    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ConfigurationError(
            f"Configuration key 'training.max_epochs' must be an integer >= 1, "
            f"got: {max_epochs}."
        )

    if not isinstance(patience, int) or patience < 1:
        raise ConfigurationError(
            f"Configuration key 'training.patience' must be an integer >= 1, "
            f"got: {patience}."
        )

    if lr_schedule not in _VALID_LR_SCHEDULES:
        raise ConfigurationError(
            f"Configuration key 'training.lr_schedule' has invalid value "
            f"'{lr_schedule}'. Must be one of: {sorted(_VALID_LR_SCHEDULES)}."
        )

    if not isinstance(random_seed, int):
        raise ConfigurationError(
            f"Configuration key 'training.random_seed' must be an integer, "
            f"got: {type(random_seed).__name__}."
        )

    return TrainingConfig(
        learning_rate=float(learning_rate),
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr_schedule=lr_schedule,
        random_seed=random_seed,
    )


def _validate_threshold(raw: dict) -> ThresholdConfig:
    percentile = _require_key(raw, "percentile", "threshold")

    if not isinstance(percentile, int) or not (1 <= percentile <= 99):
        raise ConfigurationError(
            f"Configuration key 'threshold.percentile' must be an integer in "
            f"[1, 99] inclusive, got: {percentile}."
        )

    return ThresholdConfig(percentile=percentile)


def _validate_output(raw: dict) -> OutputConfig:
    artefact_dir = _require_key(raw, "artefact_dir", "output")

    if not isinstance(artefact_dir, str) or not artefact_dir.strip():
        raise ConfigurationError(
            "Configuration key 'output.artefact_dir' must be a non-empty string."
        )

    return OutputConfig(artefact_dir=artefact_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration file.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file to load.

    Returns
    -------
    ExperimentConfig
        Fully validated experiment configuration object.

    Raises
    ------
    ConfigurationError
        If any required top-level key is missing, or if any field value is
        outside its valid range.  The exception message always names the
        offending key.
    OSError
        If ``yaml_path`` cannot be opened or read.
    """
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Expected a YAML mapping at the top level of '{yaml_path}', "
            f"got: {type(raw).__name__}."
        )

    # Validate all six required top-level keys are present before processing
    for key in _REQUIRED_TOP_LEVEL_KEYS:
        _require_key(raw, key)

    dataset = _validate_dataset(raw["dataset"])
    splits = _validate_splits(raw["splits"])
    architecture = _validate_architecture(raw["architecture"])
    training = _validate_training(raw["training"])
    threshold = _validate_threshold(raw["threshold"])
    output = _validate_output(raw["output"])

    return ExperimentConfig(
        dataset=dataset,
        splits=splits,
        architecture=architecture,
        training=training,
        threshold=threshold,
        output=output,
    )
