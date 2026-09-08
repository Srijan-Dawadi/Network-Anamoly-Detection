"""
evaluator.py
============
Evaluator — computes classification metrics and generates visualisation
artefacts (confusion matrix, ROC curve, reconstruction-error histogram)
for the network-anomaly autoencoder.
"""

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class Evaluator:
    """Compute classification metrics and write plot + JSON artefacts.

    Parameters
    ----------
    logger : logging.Logger
        Caller-supplied logger instance.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mse_scores: np.ndarray,
        threshold: float,
        run_dir: str,
    ) -> dict:
        """Compute all metrics and generate all plots.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground-truth labels (0 = normal, 1 = anomalous).
        y_pred : np.ndarray
            Binary predictions produced by the model.
        mse_scores : np.ndarray
            Per-sample reconstruction errors.
        threshold : float
            Anomaly decision threshold.
        run_dir : str
            Directory where PNG + JSON artefacts are written.

        Returns
        -------
        dict
            Metrics dictionary (also written to ``metrics.json``).

        Raises
        ------
        ValueError
            If the input array lengths do not match.
        """
        self._validate_lengths(y_true, y_pred, mse_scores)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mse_scores = np.asarray(mse_scores)

        classification = self._compute_classification_metrics(y_true, y_pred)
        auc = self._compute_auc_roc(y_true, mse_scores)

        metrics = dict(classification)
        metrics["auc_roc"] = auc

        self._plot_confusion_matrix(y_true, y_pred, run_dir)
        self._plot_roc_curve(y_true, mse_scores, run_dir)
        self._plot_re_histogram(mse_scores, y_true, threshold, run_dir)
        self._write_metrics_json(metrics, run_dir)

        self._logger.info(
            "Evaluation complete: accuracy=%.4f auc_roc=%.4f",
            metrics["accuracy"],
            metrics["auc_roc"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _validate_lengths(
        self, y_true: np.ndarray, y_pred: np.ndarray, mse_scores: np.ndarray
    ) -> None:
        lengths = {"y_true": len(y_true), "y_pred": len(y_pred), "mse_scores": len(mse_scores)}
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(
                "Length mismatch in evaluate(): got "
                + ", ".join(f"{k}={v}" for k, v in lengths.items())
                + ". All inputs must have the same length."
            )

    def _compute_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
        )

        accuracy = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )

        metrics = {
            "precision_normal": float(precision[0]),
            "recall_normal": float(recall[0]),
            "f1_normal": float(f1[0]),
            "precision_anomalous": float(precision[1]),
            "recall_anomalous": float(recall[1]),
            "f1_anomalous": float(f1[1]),
            "accuracy": accuracy,
        }
        return metrics

    def _compute_auc_roc(self, y_true: np.ndarray, mse_scores: np.ndarray) -> float:
        auc = float(roc_auc_score(y_true, mse_scores))
        if auc < 0.80:
            self._logger.warning(
                "Model may be underperforming: AUC-ROC=%.4f is below 0.80.",
                auc,
            )
        return auc

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, run_dir: str
    ) -> None:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Normal", "Anomalous"],
            yticklabels=["Normal", "Anomalous"],
            xlabel="Predicted label",
            ylabel="True label",
            title="Confusion Matrix",
        )
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        path = os.path.join(run_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        self._logger.info("Saved confusion matrix to %s", path)

    def _plot_roc_curve(
        self, y_true: np.ndarray, mse_scores: np.ndarray, run_dir: str
    ) -> None:
        fpr, tpr, _ = roc_curve(y_true, mse_scores)
        auc = roc_auc_score(y_true, mse_scores)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC Curve",
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = os.path.join(run_dir, "roc_curve.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        self._logger.info("Saved ROC curve to %s", path)

    def _plot_re_histogram(
        self,
        mse_scores: np.ndarray,
        y_true: np.ndarray,
        threshold: float,
        run_dir: str,
    ) -> None:
        normal = mse_scores[y_true == 0]
        anomalous = mse_scores[y_true == 1]

        fig, ax = plt.subplots(figsize=(6, 4))
        if len(normal) > 0:
            ax.hist(normal, bins=30, alpha=0.6, label="Normal")
        if len(anomalous) > 0:
            ax.hist(anomalous, bins=30, alpha=0.6, label="Anomalous")
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
        ax.set(
            xlabel="Reconstruction Error (MSE)",
            ylabel="Frequency",
            title="Reconstruction Error Histogram",
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        path = os.path.join(run_dir, "re_histogram.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        self._logger.info("Saved reconstruction-error histogram to %s", path)

    def _write_metrics_json(self, metrics: dict, run_dir: str) -> None:
        path = os.path.join(run_dir, "metrics.json")
        with open(path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        self._logger.info("Wrote metrics report to %s", path)
