"""
tests/unit/test_preprocessing.py
=================================
Unit tests for ``Data_Pipeline`` preprocessing functionality.

Covers:
  - ConfigurationError on invalid split ratios (zero, negative, non-summing)
  - DataLoadError when fewer than 2 normal records are present
  - Train/val splits contain only normal samples; test split contains both
  - Consistent feature dimension across all splits produced by fit_transform
  - save_artefacts + load_artefacts round-trip (transform succeeds after reload)
  - ArtifactNotFoundError when artefact files are missing
  - InferenceError raised by transform when artefacts not loaded

Validates: Requirements 2.4, 2.6
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from network_anomaly_autoencoder.config import (
    ArchitectureConfig,
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    SplitsConfig,
    ThresholdConfig,
    TrainingConfig,
)
from network_anomaly_autoencoder.data_pipeline import Data_Pipeline
from network_anomaly_autoencoder.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    DataLoadError,
    InferenceError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)


def _make_config(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> ExperimentConfig:
    """Build a minimal but fully valid ``ExperimentConfig`` pointing at *csv_path*."""
    return ExperimentConfig(
        dataset=DatasetConfig(path=csv_path, schema="nsl_kdd"),
        splits=SplitsConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=42,
        ),
        architecture=ArchitectureConfig(
            encoder_layers=[64, 32],
            bottleneck_dim=16,
            activation="relu",
            dropout_rate=0.0,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            max_epochs=10,
            patience=3,
            lr_schedule="none",
            random_seed=42,
        ),
        threshold=ThresholdConfig(percentile=95),
        output=OutputConfig(artefact_dir="/tmp/artefacts"),
    )


def _make_pipeline(config: ExperimentConfig) -> Data_Pipeline:
    return Data_Pipeline(config=config, logger=_LOGGER)


# ---------------------------------------------------------------------------
# ConfigurationError — invalid split ratios
# ---------------------------------------------------------------------------


class TestInvalidSplitRatios:
    """ConfigurationError must be raised before any data is touched."""

    def test_zero_train_ratio_raises(self, nsl_kdd_csv_path: str, nsl_kdd_df_small: pd.DataFrame) -> None:
        """train_ratio=0.0 is outside (0, 1) and must raise ConfigurationError."""
        cfg = _make_config(nsl_kdd_csv_path, train_ratio=0.0, val_ratio=0.5, test_ratio=0.5)
        pipeline = _make_pipeline(cfg)
        with pytest.raises(ConfigurationError):
            pipeline.fit_transform(nsl_kdd_df_small)

    def test_negative_ratio_raises(self, nsl_kdd_csv_path: str, nsl_kdd_df_small: pd.DataFrame) -> None:
        """A negative ratio is invalid and must raise ConfigurationError."""
        cfg = _make_config(nsl_kdd_csv_path, train_ratio=-0.3, val_ratio=0.7, test_ratio=0.6)
        pipeline = _make_pipeline(cfg)
        with pytest.raises(ConfigurationError):
            pipeline.fit_transform(nsl_kdd_df_small)

    def test_ratios_not_summing_to_one_raises(self, nsl_kdd_csv_path: str, nsl_kdd_df_small: pd.DataFrame) -> None:
        """Ratios 0.6+0.2+0.1=0.9 don't sum to 1.0 — must raise ConfigurationError."""
        cfg = _make_config(nsl_kdd_csv_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)
        pipeline = _make_pipeline(cfg)
        with pytest.raises(ConfigurationError):
            pipeline.fit_transform(nsl_kdd_df_small)


# ---------------------------------------------------------------------------
# DataLoadError — too few normal records
# ---------------------------------------------------------------------------


class TestTooFewNormalRecords:
    """DataLoadError must be raised when fewer than 2 normal rows are present."""

    def test_single_normal_row_raises(self, nsl_kdd_csv_path: str) -> None:
        """A DataFrame with only 1 normal row must cause DataLoadError."""
        # Build a tiny DataFrame: 1 normal row + 5 attack rows
        from tests.conftest import (
            NSL_KDD_NUMERICAL_COLUMNS,
            NSL_KDD_CATEGORICAL_COLUMNS,
            NSL_KDD_PROTOCOL_VALUES,
            NSL_KDD_SERVICE_VALUES,
            NSL_KDD_FLAG_VALUES,
        )
        rng = np.random.default_rng(0)
        n = 6  # 1 normal + 5 attack
        data: dict = {}
        for col in NSL_KDD_NUMERICAL_COLUMNS:
            data[col] = rng.standard_normal(n)
        data["protocol_type"] = rng.choice(NSL_KDD_PROTOCOL_VALUES, size=n)
        data["service"] = rng.choice(NSL_KDD_SERVICE_VALUES, size=n)
        data["flag"] = rng.choice(NSL_KDD_FLAG_VALUES, size=n)
        data["label"] = ["normal"] + ["neptune"] * 5
        tiny_df = pd.DataFrame(data)

        cfg = _make_config(nsl_kdd_csv_path)
        pipeline = _make_pipeline(cfg)
        with pytest.raises(DataLoadError):
            pipeline.fit_transform(tiny_df)


# ---------------------------------------------------------------------------
# Correct split label composition
# ---------------------------------------------------------------------------


class TestSplitLabelComposition:
    """Train and val splits must be all-normal; test split must contain both."""

    def test_train_val_all_normal_test_has_both(
        self, nsl_kdd_csv_path: str, nsl_kdd_df_small: pd.DataFrame
    ) -> None:
        """y_train and y_val are all 0; y_test contains 0s and 1s."""
        cfg = _make_config(nsl_kdd_csv_path)
        pipeline = _make_pipeline(cfg)
        _, _, _, y_train, y_val, y_test = pipeline.fit_transform(nsl_kdd_df_small)

        # Train and val must be exclusively normal (label 0)
        assert np.all(y_train == 0), "y_train contains non-normal samples"
        assert np.all(y_val == 0), "y_val contains non-normal samples"

        # Test set must contain both normal (0) and attack (1)
        assert 0 in y_test, "y_test has no normal samples"
        assert 1 in y_test, "y_test has no attack samples"


# ---------------------------------------------------------------------------
# Consistent feature dimension across all splits
# ---------------------------------------------------------------------------


class TestFeatureDimConsistency:
    """All three splits returned by fit_transform must share the same feature dim."""

    def test_all_splits_same_feature_dim(
        self, nsl_kdd_csv_path: str, nsl_kdd_df_small: pd.DataFrame
    ) -> None:
        cfg = _make_config(nsl_kdd_csv_path)
        pipeline = _make_pipeline(cfg)
        X_train, X_val, X_test, _, _, _ = pipeline.fit_transform(nsl_kdd_df_small)

        assert X_train.ndim == 2
        assert X_val.ndim == 2
        assert X_test.ndim == 2

        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
            f"Feature dims differ: train={X_train.shape[1]}, "
            f"val={X_val.shape[1]}, test={X_test.shape[1]}"
        )


# ---------------------------------------------------------------------------
# save_artefacts + load_artefacts round-trip
# ---------------------------------------------------------------------------


class TestArtefactRoundTrip:
    """Saving and then loading artefacts must allow transform to succeed."""

    def test_save_load_transform_succeeds(
        self,
        nsl_kdd_csv_path: str,
        nsl_kdd_df_small: pd.DataFrame,
        tmp_path,
    ) -> None:
        """fit_transform → save_artefacts → fresh pipeline → load_artefacts → transform works."""
        cfg = _make_config(nsl_kdd_csv_path)

        # Fit and save
        pipeline_fit = _make_pipeline(cfg)
        pipeline_fit.fit_transform(nsl_kdd_df_small)
        pipeline_fit.save_artefacts(str(tmp_path))

        # Fresh pipeline — load from disk then transform
        pipeline_infer = _make_pipeline(cfg)
        pipeline_infer.load_artefacts(str(tmp_path))
        result = pipeline_infer.transform(nsl_kdd_df_small)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(nsl_kdd_df_small)
        assert result.ndim == 2

    def test_load_artefacts_missing_files_raises(
        self,
        nsl_kdd_csv_path: str,
        tmp_path,
    ) -> None:
        """load_artefacts must raise ArtifactNotFoundError when pkl files are absent."""
        cfg = _make_config(nsl_kdd_csv_path)
        pipeline = _make_pipeline(cfg)
        # tmp_path is empty — no encoder.pkl / scaler.pkl
        with pytest.raises(ArtifactNotFoundError):
            pipeline.load_artefacts(str(tmp_path))


# ---------------------------------------------------------------------------
# InferenceError — transform called without fitting or loading artefacts
# ---------------------------------------------------------------------------


class TestInferenceErrorWithoutArtefacts:
    """transform on a fresh (unfitted) pipeline must raise InferenceError."""

    def test_transform_without_fit_raises(
        self, nsl_kdd_csv_path: str, nsl_kdd_df_small: pd.DataFrame
    ) -> None:
        cfg = _make_config(nsl_kdd_csv_path)
        pipeline = _make_pipeline(cfg)  # never called fit_transform or load_artefacts
        with pytest.raises(InferenceError):
            pipeline.transform(nsl_kdd_df_small)
