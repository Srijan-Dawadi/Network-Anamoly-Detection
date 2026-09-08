"""
threshold.py
============
Threshold_Estimator — computes, persists, and loads the anomaly detection
threshold derived from the MSE distribution of normal-traffic training samples.

Usage
-----
    from network_anomaly_autoencoder.threshold import Threshold_Estimator

    estimator = Threshold_Estimator(config=cfg.threshold, logger=logger)
    threshold = estimator.fit(mse_scores)
    estimator.save(run_dir)

    # Later, in inference mode:
    estimator2 = Threshold_Estimator(config=cfg.threshold, logger=logger)
    threshold = estimator2.load(artefact_dir)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from network_anomaly_autoencoder.config import ThresholdConfig
from network_anomaly_autoencoder.exceptions import ArtifactLoadError, InferenceError

# Name of the JSON file written/read by this class
_THRESHOLD_FILENAME = "threshold.json"

# All four keys that must be present in the JSON artefact
_REQUIRED_KEYS = ("threshold", "percentile", "mse_mean", "mse_std")


class Threshold_Estimator:
    """Estimates, saves, and loads the anomaly-detection MSE threshold.

    Parameters
    ----------
    config:
        :class:`~network_anomaly_autoencoder.config.ThresholdConfig` with the
        desired ``percentile`` (integer in [1, 99]).
    logger:
        A standard :class:`logging.Logger` instance.
    """

    def __init__(self, config: ThresholdConfig, logger: logging.Logger) -> None:
        self._config = config
        self._logger = logger
        self._threshold: Optional[float] = None
        self._mse_mean: Optional[float] = None
        self._mse_std: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, mse_scores: np.ndarray) -> float:
        """Compute the threshold at ``config.percentile`` of *mse_scores*.

        Stores the threshold, mean, and standard deviation of the MSE
        distribution, then logs all four values at INFO level.

        Parameters
        ----------
        mse_scores:
            1-D NumPy array of per-sample MSE values from the training set.

        Returns
        -------
        float
            The computed threshold value.
        """
        self._threshold = float(np.percentile(mse_scores, self._config.percentile))
        self._mse_mean = float(np.mean(mse_scores))
        self._mse_std = float(np.std(mse_scores))

        self._logger.info(
            "Threshold computed: threshold=%.6f  percentile=%d  "
            "mse_mean=%.6f  mse_std=%.6f",
            self._threshold,
            self._config.percentile,
            self._mse_mean,
            self._mse_std,
        )

        return self._threshold

    def save(self, run_dir: str) -> None:
        """Write threshold statistics to ``<run_dir>/threshold.json``.

        The JSON file contains exactly four keys:
        ``threshold``, ``percentile``, ``mse_mean``, ``mse_std``.

        Parameters
        ----------
        run_dir:
            Path to the timestamped artefact directory for this run.
        """
        payload = {
            "threshold": self._threshold,
            "percentile": self._config.percentile,
            "mse_mean": self._mse_mean,
            "mse_std": self._mse_std,
        }
        dest = os.path.join(run_dir, _THRESHOLD_FILENAME)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self._logger.info("Threshold artefact saved to '%s'.", dest)

    def load(self, artefact_dir: str) -> float:
        """Load the threshold from ``<artefact_dir>/threshold.json``.

        Validates that all four required fields are present and parseable
        as numbers.

        Parameters
        ----------
        artefact_dir:
            Path to the artefact directory that contains ``threshold.json``.

        Returns
        -------
        float
            The threshold value read from the file.

        Raises
        ------
        ArtifactLoadError
            If the file is missing, contains invalid JSON, or any required
            field is absent or cannot be interpreted as a number.
        """
        src = os.path.join(artefact_dir, _THRESHOLD_FILENAME)

        # -- read file -------------------------------------------------------
        try:
            with open(src, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            raise ArtifactLoadError(
                f"Threshold artefact not found: '{src}'."
            ) from None
        except json.JSONDecodeError as exc:
            raise ArtifactLoadError(
                f"Threshold artefact '{src}' contains invalid JSON: {exc}."
            ) from exc

        if not isinstance(data, dict):
            raise ArtifactLoadError(
                f"Threshold artefact '{src}' must be a JSON object, "
                f"got {type(data).__name__}."
            )

        # -- validate required keys ------------------------------------------
        missing_fields: list[str] = []
        unparseable_fields: list[str] = []

        for key in _REQUIRED_KEYS:
            if key not in data:
                missing_fields.append(key)
                continue
            try:
                float(data[key])
            except (TypeError, ValueError):
                unparseable_fields.append(key)

        problems: list[str] = []
        if missing_fields:
            problems.append(f"missing fields: {missing_fields}")
        if unparseable_fields:
            problems.append(f"unparseable fields: {unparseable_fields}")

        if problems:
            raise ArtifactLoadError(
                f"Threshold artefact '{src}' is invalid — "
                + "; ".join(problems)
                + "."
            )

        # -- populate state --------------------------------------------------
        self._threshold = float(data["threshold"])
        self._mse_mean = float(data["mse_mean"])
        self._mse_std = float(data["mse_std"])

        self._logger.info(
            "Threshold artefact loaded from '%s': threshold=%.6f  "
            "percentile=%s  mse_mean=%.6f  mse_std=%.6f",
            src,
            self._threshold,
            data["percentile"],
            self._mse_mean,
            self._mse_std,
        )

        return self._threshold

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Return the stored threshold value.

        Raises
        ------
        InferenceError
            If neither :meth:`fit` nor :meth:`load` has been called yet.
        """
        if self._threshold is None:
            raise InferenceError("Threshold not yet computed or loaded")
        return self._threshold
