"""
tests/unit/test_data_pipeline.py
=================================
Unit tests for Data_Pipeline.load_and_validate, _deduplicate, and _report_counts.

Coverage for requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

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
from network_anomaly_autoencoder.exceptions import DataLoadError, SchemaValidationError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_config(path: str, schema: str = "nsl_kdd") -> ExperimentConfig:
    """Build a minimal ExperimentConfig pointing to *path*."""
    return ExperimentConfig(
        dataset=DatasetConfig(path=path, schema=schema),
        splits=SplitsConfig(
            train_ratio=0.70,
            val_ratio=0.10,
            test_ratio=0.20,
            random_seed=42,
        ),
        architecture=ArchitectureConfig(
            encoder_layers=[128, 64, 32],
            bottleneck_dim=16,
            activation="relu",
            dropout_rate=0.2,
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            batch_size=256,
            max_epochs=100,
            patience=10,
            lr_schedule="none",
            random_seed=42,
        ),
        threshold=ThresholdConfig(percentile=95),
        output=OutputConfig(artefact_dir="runs"),
    )


def _make_pipeline(path: str, schema: str = "nsl_kdd") -> Data_Pipeline:
    """Create a Data_Pipeline with a silent logger."""
    config = _make_config(path, schema)
    logger = logging.getLogger("test_pipeline")
    logger.setLevel(logging.DEBUG)
    return Data_Pipeline(config, logger)


# ---------------------------------------------------------------------------
# Tests: DataLoadError for missing / unreadable files
# ---------------------------------------------------------------------------

class TestLoadAndValidateMissingFile:
    """Requirement 1.2 — missing/unreadable file → DataLoadError with path."""

    def test_missing_file_raises_data_load_error(self, tmp_path):
        """Non-existent path must raise DataLoadError."""
        missing = str(tmp_path / "does_not_exist.csv")
        pipeline = _make_pipeline(missing)
        with pytest.raises(DataLoadError):
            pipeline.load_and_validate()

    def test_missing_file_error_contains_path(self, tmp_path):
        """The DataLoadError message must contain the offending file path."""
        missing = str(tmp_path / "ghost_file.csv")
        pipeline = _make_pipeline(missing)
        with pytest.raises(DataLoadError, match="ghost_file.csv"):
            pipeline.load_and_validate()


# ---------------------------------------------------------------------------
# Tests: DataLoadError for empty files
# ---------------------------------------------------------------------------

class TestLoadAndValidateEmptyFile:
    """Requirement 1.6 — zero records after header → DataLoadError."""

    def test_empty_csv_raises_data_load_error(self, tmp_path):
        """A CSV file that contains only a header (no data rows) must raise DataLoadError."""
        from network_anomaly_autoencoder.schemas import nsl_kdd

        csv_file = tmp_path / "empty.csv"
        # Write only the header row, no data rows
        header = ",".join(nsl_kdd.REQUIRED_COLUMNS)
        csv_file.write_text(header + "\n", encoding="utf-8")

        pipeline = _make_pipeline(str(csv_file))
        with pytest.raises(DataLoadError, match="empty"):
            pipeline.load_and_validate()


# ---------------------------------------------------------------------------
# Tests: SchemaValidationError for missing columns
# ---------------------------------------------------------------------------

class TestLoadAndValidateSchemaMissing:
    """Requirement 1.3 — missing required columns → SchemaValidationError listing them all."""

    def _write_csv_missing_columns(
        self,
        tmp_path: Path,
        missing_cols: list[str],
    ) -> str:
        """Write a minimal NSL-KDD CSV but with *missing_cols* removed."""
        from network_anomaly_autoencoder.schemas import nsl_kdd
        import numpy as np

        rng = np.random.default_rng(0)
        n = 10
        data: dict[str, object] = {}
        for col in nsl_kdd.NUMERICAL_COLUMNS:
            data[col] = rng.standard_normal(n)
        data["protocol_type"] = ["tcp"] * n
        data["service"] = ["http"] * n
        data["flag"] = ["SF"] * n
        data["label"] = ["normal"] * n

        df = pd.DataFrame(data)
        # Drop the columns we want missing
        df = df.drop(columns=missing_cols, errors="ignore")

        path = tmp_path / "missing_cols.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_single_missing_column_raises_schema_error(self, tmp_path):
        """A single absent column must raise SchemaValidationError."""
        path = self._write_csv_missing_columns(tmp_path, ["duration"])
        pipeline = _make_pipeline(path)
        with pytest.raises(SchemaValidationError):
            pipeline.load_and_validate()

    def test_missing_column_listed_in_message(self, tmp_path):
        """The missing column name must appear in the error message."""
        path = self._write_csv_missing_columns(tmp_path, ["src_bytes"])
        pipeline = _make_pipeline(path)
        with pytest.raises(SchemaValidationError, match="src_bytes"):
            pipeline.load_and_validate()

    def test_all_missing_columns_listed_in_message(self, tmp_path):
        """ALL missing column names must appear in the error message."""
        missing = ["duration", "src_bytes", "flag"]
        path = self._write_csv_missing_columns(tmp_path, missing)
        pipeline = _make_pipeline(path)
        with pytest.raises(SchemaValidationError) as exc_info:
            pipeline.load_and_validate()
        msg = str(exc_info.value)
        for col in missing:
            assert col in msg, f"Expected '{col}' to appear in SchemaValidationError message"

    def test_missing_label_column_raises_schema_error(self, tmp_path):
        """Missing the label column must raise SchemaValidationError."""
        path = self._write_csv_missing_columns(tmp_path, ["label"])
        pipeline = _make_pipeline(path)
        with pytest.raises(SchemaValidationError, match="label"):
            pipeline.load_and_validate()


# ---------------------------------------------------------------------------
# Tests: Successful load on valid synthetic NSL-KDD CSV
# ---------------------------------------------------------------------------

class TestLoadAndValidateSuccess:
    """Requirements 1.1, 1.4, 1.5 — successful load returns correct DataFrame."""

    def test_successful_load_returns_dataframe(self, nsl_kdd_csv_path):
        """A valid NSL-KDD CSV must return a non-empty DataFrame."""
        pipeline = _make_pipeline(nsl_kdd_csv_path)
        df = pipeline.load_and_validate()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_returned_df_has_all_required_columns(self, nsl_kdd_csv_path):
        """Returned DataFrame must contain all REQUIRED_COLUMNS."""
        from network_anomaly_autoencoder.schemas import nsl_kdd

        pipeline = _make_pipeline(nsl_kdd_csv_path)
        df = pipeline.load_and_validate()
        for col in nsl_kdd.REQUIRED_COLUMNS:
            assert col in df.columns, f"Column '{col}' missing from returned DataFrame"

    def test_load_logs_row_count(self, nsl_kdd_csv_path, caplog):
        """The pipeline must log a message containing row count information."""
        pipeline = _make_pipeline(nsl_kdd_csv_path)
        with caplog.at_level(logging.INFO):
            pipeline.load_and_validate()
        # Verify that at least one INFO message mentioning loaded rows was emitted
        assert any("Loaded CSV" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: _deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    """Requirement 1.5 — duplicate rows are removed and count is logged."""

    def _make_df_with_dupes(self, n_unique: int = 10, n_dupes: int = 5) -> pd.DataFrame:
        """Return a small synthetic NSL-KDD DataFrame with injected duplicates."""
        from network_anomaly_autoencoder.schemas import nsl_kdd
        import numpy as np

        rng = np.random.default_rng(0)
        data: dict[str, object] = {}
        for col in nsl_kdd.NUMERICAL_COLUMNS:
            data[col] = list(rng.standard_normal(n_unique)) * 1
        data["protocol_type"] = ["tcp"] * n_unique
        data["service"] = ["http"] * n_unique
        data["flag"] = ["SF"] * n_unique
        data["label"] = ["normal"] * n_unique

        df_unique = pd.DataFrame(data)
        # Inject duplicates by repeating the first n_dupes rows
        df_with_dupes = pd.concat(
            [df_unique, df_unique.iloc[:n_dupes]], ignore_index=True
        )
        return df_with_dupes

    def test_duplicates_are_removed(self, tmp_path):
        """After _deduplicate, no duplicate rows should remain."""
        config = _make_config(str(tmp_path / "dummy.csv"))
        logger = logging.getLogger("test")
        pipeline = Data_Pipeline(config, logger)

        df_with_dupes = self._make_df_with_dupes(n_unique=10, n_dupes=5)
        df_deduped = pipeline._deduplicate(df_with_dupes)

        assert len(df_deduped) == 10

    def test_deduplication_logs_removed_count(self, tmp_path, caplog):
        """_deduplicate must log how many rows were removed at INFO level."""
        config = _make_config(str(tmp_path / "dummy.csv"))
        logger = logging.getLogger("test_dedup_log")
        pipeline = Data_Pipeline(config, logger)

        df_with_dupes = self._make_df_with_dupes(n_unique=10, n_dupes=3)
        with caplog.at_level(logging.INFO):
            pipeline._deduplicate(df_with_dupes)

        assert any(
            "dedup" in r.message.lower() or "duplicate" in r.message.lower()
            for r in caplog.records
        )

    def test_no_dupes_unchanged(self, tmp_path):
        """A DataFrame with no duplicates should be returned unchanged."""
        config = _make_config(str(tmp_path / "dummy.csv"))
        logger = logging.getLogger("test")
        pipeline = Data_Pipeline(config, logger)

        df_no_dupes = self._make_df_with_dupes(n_unique=10, n_dupes=0)
        df_result = pipeline._deduplicate(df_no_dupes)
        assert len(df_result) == 10


# ---------------------------------------------------------------------------
# Tests: _report_counts
# ---------------------------------------------------------------------------

class TestReportCounts:
    """Requirement 1.4 — total, normal, and attack counts are logged."""

    def _make_simple_df(self, n_normal: int, n_attack: int) -> pd.DataFrame:
        """Return a tiny DataFrame with the label column only."""
        return pd.DataFrame(
            {"label": ["normal"] * n_normal + ["neptune"] * n_attack}
        )

    def test_report_counts_logs_info(self, tmp_path, caplog):
        """_report_counts must emit at least one INFO log message."""
        config = _make_config(str(tmp_path / "dummy.csv"))
        logger = logging.getLogger("test_report")
        pipeline = Data_Pipeline(config, logger)

        df = self._make_simple_df(n_normal=80, n_attack=20)
        with caplog.at_level(logging.INFO):
            pipeline._report_counts(df)

        assert any(r.levelno == logging.INFO for r in caplog.records)

    def test_report_counts_mentions_counts(self, tmp_path, caplog):
        """The log message must include total, normal, and attack counts."""
        config = _make_config(str(tmp_path / "dummy.csv"))
        logger = logging.getLogger("test_counts_vals")
        pipeline = Data_Pipeline(config, logger)

        df = self._make_simple_df(n_normal=80, n_attack=20)
        with caplog.at_level(logging.INFO):
            pipeline._report_counts(df)

        # Gather all log text
        log_text = " ".join(r.message for r in caplog.records)
        # All three numbers should appear somewhere in the logs
        assert "100" in log_text, "Expected total count 100 in log"
        assert "80" in log_text, "Expected normal count 80 in log"
        assert "20" in log_text, "Expected attack count 20 in log"


# ---------------------------------------------------------------------------
# Tests: Dynamic schema selection
# ---------------------------------------------------------------------------

class TestSchemaSelection:
    """__init__ must load the correct schema module for each dataset name."""

    def test_nsl_kdd_schema_loaded(self, tmp_path):
        """nsl_kdd schema must expose FEATURE_COUNT == 41."""
        pipeline = _make_pipeline(str(tmp_path / "x.csv"), schema="nsl_kdd")
        assert pipeline._schema.FEATURE_COUNT == 41

    def test_cicids2017_schema_loaded(self, tmp_path):
        """cicids2017 schema must expose FEATURE_COUNT == 80."""
        pipeline = _make_pipeline(str(tmp_path / "x.csv"), schema="cicids2017")
        assert pipeline._schema.FEATURE_COUNT == 80

    def test_unsw_nb15_schema_loaded(self, tmp_path):
        """unsw_nb15 schema must expose FEATURE_COUNT == 49."""
        pipeline = _make_pipeline(str(tmp_path / "x.csv"), schema="unsw_nb15")
        assert pipeline._schema.FEATURE_COUNT == 49
