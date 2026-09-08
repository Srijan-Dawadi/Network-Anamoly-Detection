"""
experiment_tracker.py
=====================
Experiment_Tracker — orchestrates training and inference runs, wires together
the data pipeline, model, threshold estimator, and evaluator, and manages
versioned artefact directories.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from network_anomaly_autoencoder.config import ExperimentConfig, load_config
from network_anomaly_autoencoder.data_pipeline import Data_Pipeline
from network_anomaly_autoencoder.evaluator import Evaluator
from network_anomaly_autoencoder.exceptions import (
    ArtifactNotFoundError,
    ConfigurationError,
    NetworkAnomalyBaseError,
)
from network_anomaly_autoencoder.model import Autoencoder_Model
from network_anomaly_autoencoder.threshold import Threshold_Estimator
from network_anomaly_autoencoder.utils.logging_utils import get_logger

# Names of the four inference-critical artefact files
_CRITICAL_ARTEFACTS = (
    "model_weights.keras",
    "scaler.pkl",
    "encoder.pkl",
    "threshold.json",
)


class Experiment_Tracker:
    """Coordinates a full training or inference experiment from a YAML config.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML experiment configuration file.
    """

    def __init__(self, yaml_path: str) -> None:
        self._yaml_path = yaml_path
        self._config: Optional[ExperimentConfig] = None
        self._logger: Optional[logging.Logger] = None
        self._run_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> ExperimentConfig:
        """Load config, create timestamped run dir, configure logger, copy YAML.

        Returns
        -------
        ExperimentConfig
            The fully validated configuration object.

        Raises
        ------
        ConfigurationError
            If any required YAML key is missing or a value is invalid.
        """
        config = load_config(self._yaml_path)
        self._config = config

        self._run_dir = self._create_run_dir()

        # Logger writes to both console and a per-run log file.
        log_file = os.path.join(self._run_dir, "experiment.log")
        self._logger = get_logger("network_anomaly_autoencoder", log_file=log_file)

        # Byte-for-byte copy of the input YAML, preserving its filename.
        dest_config = os.path.join(self._run_dir, os.path.basename(self._yaml_path))
        shutil.copyfile(self._yaml_path, dest_config)

        self._log_git_hash()
        self._logger.info("Run directory created: %s", self._run_dir)
        self._logger.info("Experiment setup complete for %s", self._yaml_path)

        return config

    def _create_run_dir(self) -> str:
        if self._config is None:
            raise ConfigurationError(
                "Cannot create run directory before configuration is loaded."
            )
        base = self._config.output.artefact_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _log_git_hash(self) -> None:
        """Log the current Git commit hash, or ``"git-unavailable"``."""
        if self._logger is None:
            return
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                self._logger.info("Git commit hash: %s", commit)
            else:
                self._logger.info("Git commit hash: git-unavailable")
        except (OSError, subprocess.SubprocessError):
            self._logger.info("Git commit hash: git-unavailable")

    def _record_timing(self, elapsed_secs: float) -> None:
        """Write training wall-clock time into the run's metrics.json."""
        if self._run_dir is None:
            return
        metrics_path = os.path.join(self._run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            return
        import json

        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        metrics["training_time_secs"] = round(elapsed_secs, 2)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        self._logger.info(
            "Training wall-clock time: %.2f seconds", elapsed_secs
        )

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def run_training(self) -> None:
        """Run the full training pipeline and write all artefacts to run_dir.

        Orchestrates: load → preprocess → build → train → threshold →
        evaluate → save artefacts.
        """
        if self._config is None or self._logger is None or self._run_dir is None:
            self._logger = get_logger("network_anomaly_autoencoder")
            self.setup()

        config = self._config
        logger = self._logger
        run_dir = self._run_dir
        assert config is not None and logger is not None and run_dir is not None

        start_time = time.time()
        try:
            # ---- Phase: data loading -----------------------------------
            logger.info("Phase: data loading started")
            pipeline = Data_Pipeline(config, logger)
            df = pipeline.load_and_validate()
            logger.info("Phase: data loading complete")

            # ---- Phase: preprocessing ----------------------------------
            logger.info("Phase: preprocessing started")
            X_train, X_val, X_test, y_train, y_val, y_test = pipeline.fit_transform(df)
            pipeline.save_artefacts(run_dir)
            logger.info(
                "Phase: preprocessing complete (feature_dim=%d)",
                X_train.shape[1],
            )

            # ---- Phase: model initialisation ---------------------------
            logger.info("Phase: model initialisation started")
            model = Autoencoder_Model(config.architecture, config.training, logger)
            model.build(X_train.shape[1])
            logger.info("Phase: model initialisation complete")

            # ---- Phase: training ---------------------------------------
            logger.info("Phase: training started")
            model.fit(X_train, X_val, run_dir)
            logger.info("Phase: training complete")

            # ---- Phase: threshold estimation ---------------------------
            logger.info("Phase: threshold estimation started")
            _, train_mse = model.reconstruct(X_train)
            threshold_estimator = Threshold_Estimator(config.threshold, logger)
            threshold = threshold_estimator.fit(train_mse)
            threshold_estimator.save(run_dir)
            logger.info("Phase: threshold estimation complete")

            # ---- Phase: evaluation -------------------------------------
            logger.info("Phase: evaluation started")
            y_pred, mse_scores = model.predict(X_test, threshold)
            evaluator = Evaluator(logger)
            evaluator.evaluate(y_test, y_pred, mse_scores, threshold, run_dir)
            logger.info("Phase: evaluation complete")

            elapsed = time.time() - start_time
            self._record_timing(elapsed)

            logger.info("Training run completed successfully in %.2fs", elapsed)
        except NetworkAnomalyBaseError:
            logger.exception("Training run failed with a project error")
            raise
        except Exception:
            logger.exception("Training run failed with an unexpected error")
            raise

    # ------------------------------------------------------------------
    # Inference orchestration
    # ------------------------------------------------------------------

    def run_inference(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on a raw CSV file using a previously saved run dir.

        Parameters
        ----------
        csv_path : str
            Path to a CSV file containing records to classify.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``(predictions, mse_scores)`` for the input records.

        Raises
        ------
        ArtifactNotFoundError
            If any of the four inference-critical artefact files is absent.
        """
        if self._config is None or self._logger is None or self._run_dir is None:
            self._logger = get_logger("network_anomaly_autoencoder")
            self.setup()

        config = self._config
        logger = self._logger
        run_dir = self._run_dir
        assert config is not None and logger is not None and run_dir is not None

        logger.info("Phase: inference started")

        # All-or-nothing: verify all four critical artefacts exist first.
        missing = [
            name
            for name in _CRITICAL_ARTEFACTS
            if not os.path.isfile(os.path.join(run_dir, name))
        ]
        if missing:
            raise ArtifactNotFoundError(
                "Cannot run inference: missing artefact file(s) in "
                f"'{run_dir}': {missing}. No components were loaded."
            )

        # Inference operates on the user-supplied CSV, not the training file.
        import dataclasses

        infer_config = dataclasses.replace(
            config, dataset=dataclasses.replace(config.dataset, path=csv_path)
        )
        pipeline = Data_Pipeline(infer_config, logger)
        df = pipeline.load_and_validate()
        pipeline.load_artefacts(run_dir)
        X_new = pipeline.transform(df)
        model = Autoencoder_Model(config.architecture, config.training, logger)
        model.build(X_new.shape[1])
        model.load_weights(os.path.join(run_dir, "model_weights.keras"))

        threshold_estimator = Threshold_Estimator(config.threshold, logger)
        threshold = threshold_estimator.load(run_dir)

        predictions, mse_scores = model.predict(X_new, threshold)

        logger.info("Phase: inference complete (%d records)", X_new.shape[0])
        return predictions, mse_scores
