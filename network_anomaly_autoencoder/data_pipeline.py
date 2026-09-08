"""
data_pipeline.py
================
Data loading, validation, preprocessing, and artefact persistence for the
network-anomaly-autoencoder pipeline.

The ``Data_Pipeline`` class encapsulates two operating modes:

* **Training mode** — ``load_and_validate`` → ``fit_transform`` → ``save_artefacts``
* **Inference mode** — ``load_and_validate`` → ``load_artefacts`` → ``transform``

All public methods raise project-specific exceptions (never plain ``ValueError``
or bare ``OSError``) so callers can discriminate failure modes cleanly.
"""

from __future__ import annotations

import importlib
import logging
import os
import types
from typing import TYPE_CHECKING

import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from network_anomaly_autoencoder.config import ExperimentConfig
from network_anomaly_autoencoder.exceptions import (
    DataLoadError,
    SchemaValidationError,
    ConfigurationError,
    InferenceError,
    ArtifactNotFoundError,
)

# Mapping from schema name string to fully-qualified module path
_SCHEMA_MODULE_MAP: dict[str, str] = {
    "nsl_kdd": "network_anomaly_autoencoder.schemas.nsl_kdd",
    "cicids2017": "network_anomaly_autoencoder.schemas.cicids2017",
    "unsw_nb15": "network_anomaly_autoencoder.schemas.unsw_nb15",
}


class Data_Pipeline:
    """End-to-end data pipeline: load → validate → preprocess → split → persist.

    Parameters
    ----------
    config:
        Fully validated ``ExperimentConfig`` instance.
    logger:
        Logger to use for all INFO/DEBUG/WARNING messages emitted by this class.
    """

    def __init__(self, config: ExperimentConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger

        # Dynamically import the schema module based on the configured schema name
        schema_name = config.dataset.schema
        module_path = _SCHEMA_MODULE_MAP[schema_name]  # KeyError impossible — config validates this
        self._schema: types.ModuleType = importlib.import_module(module_path)

        # Artefact state (populated by fit_transform / load_artefacts)
        self._ohe = None   # sklearn OneHotEncoder, set after fit_transform or load_artefacts
        self._scaler = None  # sklearn StandardScaler, set after fit_transform or load_artefacts

    # ------------------------------------------------------------------
    # Public — loading and validation
    # ------------------------------------------------------------------

    def load_and_validate(self) -> pd.DataFrame:
        """Load the CSV from ``config.dataset.path``, verify schema, deduplicate,
        and report label-based counts.

        Returns
        -------
        pd.DataFrame
            DataFrame containing **at minimum** all ``REQUIRED_COLUMNS`` defined
            in the active schema.  Additional columns present in the file are
            preserved but not required.

        Raises
        ------
        DataLoadError
            * If the file does not exist or cannot be read (``FileNotFoundError``
              or ``PermissionError``).
            * If the file is readable but contains zero data rows after the
              header.
        SchemaValidationError
            If one or more ``REQUIRED_COLUMNS`` are absent from the loaded
            DataFrame.  All missing column names are included in the error
            message.
        """
        path = self._config.dataset.path

        # --- 1. Load the CSV ---------------------------------------------------
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise DataLoadError(
                f"Dataset file not found: {path}"
            ) from None
        except PermissionError:
            raise DataLoadError(
                f"Permission denied when reading dataset file: {path}"
            ) from None
        except pd.errors.EmptyDataError:
            raise DataLoadError(
                f"Dataset file contains no parseable data: {path}"
            ) from None

        self._logger.info("Loaded CSV from '%s': %d rows, %d columns.", path, len(df), len(df.columns))

        # --- 2. Check for empty file -------------------------------------------
        if len(df) == 0:
            raise DataLoadError(
                f"Dataset file is empty (zero records after header): {path}"
            )

        # --- 3. Schema validation -----------------------------------------------
        required = self._schema.REQUIRED_COLUMNS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise SchemaValidationError(
                f"Schema validation failed for '{path}'. "
                f"Missing columns ({len(missing)}): {missing}"
            )

        self._logger.info("Schema validation passed — all %d required columns present.", len(required))

        # --- 4. Deduplicate ------------------------------------------------------
        df = self._deduplicate(df)

        # --- 5. Report counts ---------------------------------------------------
        self._report_counts(df)

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that are exact duplicates across all ``REQUIRED_COLUMNS``.

        Parameters
        ----------
        df:
            DataFrame to deduplicate (not modified in place).

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicate rows removed.
        """
        before = len(df)
        df_deduped = df.drop_duplicates(subset=self._schema.REQUIRED_COLUMNS)
        removed = before - len(df_deduped)
        self._logger.info(
            "Deduplication: removed %d duplicate row(s); %d row(s) remaining.",
            removed,
            len(df_deduped),
        )
        return df_deduped

    def _report_counts(self, df: pd.DataFrame) -> None:
        """Log total, normal, and attack record counts at INFO level.

        Parameters
        ----------
        df:
            Deduplicated DataFrame to report counts for.
        """
        label_col = self._schema.LABEL_COLUMN
        normal_label = self._schema.NORMAL_LABEL

        total = len(df)
        normal_count = int((df[label_col] == normal_label).sum())
        attack_count = total - normal_count

        self._logger.info(
            "Record counts — total: %d | normal: %d | attack: %d.",
            total,
            normal_count,
            attack_count,
        )

    # ------------------------------------------------------------------
    # Public — preprocessing and splitting  (Task 6.1 — NOT YET IMPLEMENTED)
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One-hot encode categoricals, StandardScaler-normalise numericals,
        and split into train/val/test sets.

        Returns
        -------
        tuple
            ``(X_train_normal, X_val_normal, X_test_mixed, y_train, y_val, y_test)``
            as numpy arrays.

        Raises
        ------
        ConfigurationError
            If split ratios are invalid.
        DataLoadError
            If fewer than 2 normal records are present after splitting.
        """
        # 1. Validate split ratios first
        self._validate_split_ratios()

        label_col = self._schema.LABEL_COLUMN
        normal_label = self._schema.NORMAL_LABEL
        cat_cols = self._schema.CATEGORICAL_COLUMNS
        num_cols = self._schema.NUMERICAL_COLUMNS
        binary_map = self._schema.BINARY_LABEL_MAP

        # 2. Separate normal vs attack rows
        mask_normal = df[label_col] == normal_label
        df_normal = df[mask_normal].copy()
        df_attack = df[~mask_normal].copy()

        self._logger.info(
            "fit_transform: %d normal rows, %d attack rows.",
            len(df_normal),
            len(df_attack),
        )

        if len(df_normal) < 2:
            raise DataLoadError(
                "fit_transform requires at least 2 normal records to create a "
                f"train/val split, but only {len(df_normal)} normal record(s) found."
            )

        # 3. Create binary labels (default to 1 for unknown attack sub-types)
        y_all = df[label_col].map(binary_map).fillna(1).astype(int).to_numpy()

        # 4. Split normal rows into raw train / validation frames FIRST so the
        #    fitted encoder + scaler are derived exclusively from the training
        #    split (Property 6: training features are exactly standard-scaled).
        train_ratio = self._config.splits.train_ratio
        val_ratio = self._config.splits.val_ratio
        random_seed = self._config.splits.random_seed

        # train_size within the normal pool
        train_fraction = train_ratio / (train_ratio + val_ratio)

        df_train, df_val = train_test_split(
            df_normal,
            train_size=train_fraction,
            random_state=random_seed,
            shuffle=True,
        )

        # 5. Fit OHE on categorical columns of the training split only
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(df_train[cat_cols])
        self._ohe = ohe

        # 6. Fit StandardScaler on numerical columns of the training split only
        scaler = StandardScaler()
        scaler.fit(df_train[num_cols])
        self._scaler = scaler

        # 7. Apply OHE + Scaler to ALL data (train, val, normal+attack test)
        def _encode_df(frame: pd.DataFrame) -> np.ndarray:
            ohe_part = self._ohe.transform(frame[cat_cols])
            num_part = self._scaler.transform(frame[num_cols])
            return np.concatenate([ohe_part, num_part], axis=1)

        X_train_normal = _encode_df(df_train)
        X_val_normal = _encode_df(df_val)
        X_test_mixed = _encode_df(df)          # all rows, for test set

        y_train = np.zeros(len(X_train_normal), dtype=int)
        y_val = np.zeros(len(X_val_normal), dtype=int)
        y_test = y_all

        self._logger.info(
            "fit_transform complete: X_train=%s, X_val=%s, X_test=%s, feature_dim=%d.",
            X_train_normal.shape,
            X_val_normal.shape,
            X_test_mixed.shape,
            X_train_normal.shape[1],
        )

        return X_train_normal, X_val_normal, X_test_mixed, y_train, y_val, y_test

    def _validate_split_ratios(self) -> None:
        """Validate that split ratios in the config are valid.

        Raises
        ------
        ConfigurationError
            If any ratio is ≤ 0.0 or ≥ 1.0, or if the three ratios do not
            sum to 1.0 (within a tolerance of 1e-6).
        """
        splits = self._config.splits
        ratios = {
            "train_ratio": splits.train_ratio,
            "val_ratio": splits.val_ratio,
            "test_ratio": splits.test_ratio,
        }

        for name, value in ratios.items():
            if value <= 0.0 or value >= 1.0:
                raise ConfigurationError(
                    f"Split ratio '{name}' must be in the exclusive range (0.0, 1.0), "
                    f"got: {value}."
                )

        total = splits.train_ratio + splits.val_ratio + splits.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"Split ratios 'train_ratio', 'val_ratio', and 'test_ratio' must sum "
                f"to 1.0, but they sum to {total:.8f}."
            )

    def save_artefacts(self, run_dir: str) -> None:
        """Persist the fitted OHE and scaler to *run_dir* as ``.pkl`` files.

        Parameters
        ----------
        run_dir:
            Directory path where ``encoder.pkl`` and ``scaler.pkl`` will be written.

        Raises
        ------
        RuntimeError
            If ``fit_transform`` has not been called (artefacts are ``None``).
        """
        encoder_path = os.path.join(run_dir, "encoder.pkl")
        scaler_path = os.path.join(run_dir, "scaler.pkl")

        joblib.dump(self._ohe, encoder_path)
        self._logger.info("Saved OHE encoder to '%s'.", encoder_path)

        joblib.dump(self._scaler, scaler_path)
        self._logger.info("Saved StandardScaler to '%s'.", scaler_path)

    def load_artefacts(self, artefact_dir: str) -> None:
        """Load a previously fitted OHE and scaler from *artefact_dir*.

        Parameters
        ----------
        artefact_dir:
            Directory from which ``encoder.pkl`` and ``scaler.pkl`` are loaded.

        Raises
        ------
        ArtifactNotFoundError
            If either ``encoder.pkl`` or ``scaler.pkl`` is absent from the
            directory.
        """
        encoder_path = os.path.join(artefact_dir, "encoder.pkl")
        scaler_path = os.path.join(artefact_dir, "scaler.pkl")

        missing = [p for p in (encoder_path, scaler_path) if not os.path.isfile(p)]
        if missing:
            raise ArtifactNotFoundError(
                f"Required artefact file(s) not found: {missing}"
            )

        self._ohe = joblib.load(encoder_path)
        self._logger.info("Loaded OHE encoder from '%s'.", encoder_path)

        self._scaler = joblib.load(scaler_path)
        self._logger.info("Loaded StandardScaler from '%s'.", scaler_path)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the loaded encoder and scaler to new data.

        Parameters
        ----------
        df:
            DataFrame containing the same columns used during fitting.

        Returns
        -------
        np.ndarray
            Preprocessed feature matrix with OHE-encoded categoricals concatenated
            with scaled numerical features.

        Raises
        ------
        InferenceError
            If ``load_artefacts`` (or ``fit_transform``) has not been called yet,
            leaving ``self._ohe`` or ``self._scaler`` as ``None``.
        """
        if self._ohe is None or self._scaler is None:
            raise InferenceError(
                "Artefacts not loaded. Call 'fit_transform' (training mode) or "
                "'load_artefacts' (inference mode) before calling 'transform'."
            )

        cat_cols = self._schema.CATEGORICAL_COLUMNS
        num_cols = self._schema.NUMERICAL_COLUMNS

        ohe_part = self._ohe.transform(df[cat_cols])
        num_part = self._scaler.transform(df[num_cols])

        return np.concatenate([ohe_part, num_part], axis=1)
