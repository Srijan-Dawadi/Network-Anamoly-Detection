"""
model.py
========
Autoencoder_Model — symmetric encoder-decoder neural network for
network-traffic anomaly detection.

The model is trained exclusively on normal traffic and at inference time
flags records whose reconstruction error exceeds a learned threshold.
"""

import logging
import os
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from network_anomaly_autoencoder.config import ArchitectureConfig, TrainingConfig
from network_anomaly_autoencoder.exceptions import (
    ArtifactWriteError,
    ConfigurationError,
    DataLoadError,
    InferenceError,
)
from network_anomaly_autoencoder.utils.seed_utils import set_global_seeds


def _is_path_writable(path: str) -> tuple[bool, str]:
    """Return ``(ok, error)`` indicating whether *path* is writable.

    If the directory exists, verify we can create a temporary probe file.
    Otherwise, walk up the tree to find the nearest existing parent and
    verify it is writable (create it if needed).
    """
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f"._write_probe_{os.getpid()}.tmp")
        with open(probe, "w") as fh:
            fh.write("probe")
        os.remove(probe)
        return True, ""
    except OSError as exc:
        return False, str(exc)


class SeedLoggingMonitor(keras.callbacks.Callback):
    """Log per-epoch train/val loss and final epoch count at DEBUG/INFO."""

    def __init__(self, logger: logging.Logger) -> None:
        super().__init__()
        self._logger = logger

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        self._logger.debug(
            "Epoch %d: train_loss=%.6f val_loss=%.6f",
            epoch + 1,
            float(logs.get("loss", float("nan"))),
            float(logs.get("val_loss", float("nan"))),
        )


class Autoencoder_Model:
    """Symmetric autoencoder for unsupervised network-anomaly detection.

    Parameters
    ----------
    arch_config : ArchitectureConfig
        Architecture hyperparameters (layer sizes, bottleneck, activation,
        dropout rate).
    train_config : TrainingConfig
        Training hyperparameters (learning rate, batch size, epochs,
        patience, LR schedule, random seed).
    logger : logging.Logger
        Caller-supplied logger instance.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        train_config: TrainingConfig,
        logger: logging.Logger,
    ) -> None:
        self._arch_config = arch_config
        self._train_config = train_config
        self._logger = logger
        self._model: Optional[keras.Model] = None
        self._input_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_seeds(self) -> None:
        """Set all global random seeds for reproducibility."""
        set_global_seeds(self._train_config.random_seed)

    def _validate_input_dim(self, X: np.ndarray, caller: str) -> None:
        """Raise ``InferenceError`` if ``X`` feature count mismatches training dim."""
        if self._input_dim is None:
            raise InferenceError(
                f"Cannot call {caller}(): input dimension is not set. "
                "Call build(input_dim) first."
            )
        actual = X.shape[1]
        if actual != self._input_dim:
            raise InferenceError(
                f"Feature dimensionality mismatch in {caller}(): expected "
                f"{self._input_dim} features but got {actual}. Did you "
                "preprocess the data with the fitted encoder/scaler?"
            )

    def _build_lr_schedule(
        self,
    ) -> Optional[keras.optimizers.schedules.LearningRateSchedule]:
        """Return a Keras LR schedule or ``None`` when schedule is ``"none"``."""
        lr = self._train_config.learning_rate
        schedule = self._train_config.lr_schedule

        if schedule == "none":
            return None
        if schedule == "step_decay":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=1000,
                decay_rate=0.9,
            )
        if schedule == "cosine_annealing":
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=1000,
            )
        # Unreachable if config is validated, but kept for safety
        raise ConfigurationError(
            f"Unknown lr_schedule value: '{schedule}'. "
            "Must be one of: 'none', 'step_decay', 'cosine_annealing'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, input_dim: int) -> None:
        """Construct and compile the symmetric encoder-decoder graph.

        Architecture
        ------------
        Input(input_dim)
          → [Dense(encoder_layers[i], activation) → BatchNorm
             → Dropout(dropout_rate)?] × n          # encoder
          → Dense(bottleneck_dim, activation)        # bottleneck
          → [Dense(encoder_layers[-1-i], activation) → BatchNorm
             → Dropout(dropout_rate)?] × n          # decoder (mirrored)
          → Dense(input_dim, activation='linear')    # reconstruction

        Parameters
        ----------
        input_dim : int
            Number of input features (post-preprocessing dimensionality).

        Raises
        ------
        ConfigurationError
            If any architecture parameter is outside its valid range.
        """
        arch = self._arch_config

        # ---- Validate architecture parameters --------------------------
        for i, size in enumerate(arch.encoder_layers):
            if size < 1:
                raise ConfigurationError(
                    f"Invalid parameter 'encoder_layers[{i}]': value {size} is "
                    "less than 1. All encoder layer sizes must be >= 1."
                )

        if arch.bottleneck_dim < 1:
            raise ConfigurationError(
                f"Invalid parameter 'bottleneck_dim': value {arch.bottleneck_dim} "
                "is less than 1. bottleneck_dim must be >= 1."
            )
        min_encoder = min(arch.encoder_layers)
        if arch.bottleneck_dim >= min_encoder:
            raise ConfigurationError(
                f"Invalid parameter 'bottleneck_dim': value {arch.bottleneck_dim} "
                f"must be strictly less than the smallest encoder layer size "
                f"({min_encoder})."
            )

        if arch.dropout_rate < 0.0 or arch.dropout_rate >= 1.0:
            raise ConfigurationError(
                f"Invalid parameter 'dropout_rate': value {arch.dropout_rate} "
                "is outside the valid range [0.0, 1.0)."
            )

        # ---- Seed before weight initialisation -------------------------
        self._set_seeds()

        # ---- Build graph -----------------------------------------------
        inputs = keras.Input(shape=(input_dim,), name="input")
        x = inputs

        # Encoder
        for i, units in enumerate(arch.encoder_layers):
            x = layers.Dense(units, activation=arch.activation, name=f"enc_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
            if arch.dropout_rate > 0:
                x = layers.Dropout(arch.dropout_rate, name=f"enc_dropout_{i}")(x)

        # Bottleneck
        x = layers.Dense(arch.bottleneck_dim, activation=arch.activation, name="bottleneck")(x)

        # Decoder (symmetric mirror of encoder)
        for i, units in enumerate(reversed(arch.encoder_layers)):
            x = layers.Dense(units, activation=arch.activation, name=f"dec_dense_{i}")(x)
            x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
            if arch.dropout_rate > 0:
                x = layers.Dropout(arch.dropout_rate, name=f"dec_dropout_{i}")(x)

        # Output layer (linear activation for reconstruction)
        outputs = layers.Dense(input_dim, activation="linear", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")

        # ---- Compile ---------------------------------------------------
        lr_schedule = self._build_lr_schedule()
        learning_rate = lr_schedule if lr_schedule is not None else self._train_config.learning_rate
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mse")

        self._model = model
        self._input_dim = input_dim

        self._logger.info(
            "Autoencoder built: input_dim=%d, encoder_layers=%s, "
            "bottleneck_dim=%d, dropout_rate=%.3f",
            input_dim,
            arch.encoder_layers,
            arch.bottleneck_dim,
            arch.dropout_rate,
        )

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        run_dir: str,
    ) -> keras.callbacks.History:
        """Train on normal-traffic ``X_train``, monitoring ``X_val``.

        Parameters
        ----------
        X_train : np.ndarray
            Normal-traffic training feature matrix.
        X_val : np.ndarray
            Normal-traffic validation feature matrix.
        run_dir : str
            Artefact directory where best weights will be saved.

        Returns
        -------
        keras.callbacks.History
            Keras training history object.

        Raises
        ------
        DataLoadError
            If ``X_train`` is empty.
        ArtifactWriteError
            If the weights cannot be written to ``run_dir``.
        InferenceError
            If the model has not been built before calling ``fit``.
        """
        if self._model is None:
            raise InferenceError(
                "Cannot call fit(): the Autoencoder model has not been built. "
                "Call build(input_dim) first."
            )

        train = self._train_config

        # ---- Validate training parameters -------------------------------
        if train.learning_rate <= 0.0:
            raise ConfigurationError(
                f"Invalid parameter 'learning_rate': value {train.learning_rate} "
                "is not > 0.0."
            )
        if train.batch_size < 1:
            raise ConfigurationError(
                f"Invalid parameter 'batch_size': value {train.batch_size} "
                "is less than 1."
            )
        if train.max_epochs < 1:
            raise ConfigurationError(
                f"Invalid parameter 'max_epochs': value {train.max_epochs} "
                "is less than 1."
            )
        if train.patience < 1:
            raise ConfigurationError(
                f"Invalid parameter 'patience': value {train.patience} "
                "is less than 1."
            )

        if X_train is None or len(X_train) == 0:
            raise DataLoadError(
                "Cannot train: the training split contains zero normal-traffic "
                "samples. Provide at least one normal-traffic record."
            )

        is_writable, err = _is_path_writable(run_dir)
        if not is_writable:
            raise ArtifactWriteError(
                f"The artefact directory '{run_dir}' is not writable: {err}"
            )

        self._validate_input_dim(X_train, caller="fit")

        weights_path = os.path.join(run_dir, "model_weights.keras")

        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self._train_config.patience,
                restore_best_weights=True,
                verbose=0,
            ),
            SeedLoggingMonitor(self._logger),
        ]

        history = self._model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=self._train_config.max_epochs,
            batch_size=self._train_config.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        self.save_weights(weights_path)
        self._logger.info(
            "Training complete. Best weights saved to %s. final_train_loss=%.6f "
            "final_val_loss=%.6f",
            weights_path,
            float(history.history["loss"][-1]),
            float(history.history["val_loss"][-1]),
        )
        return history

    def reconstruct(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X_reconstructed, per_sample_mse)``.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape ``(n, input_dim)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X_hat, mse)`` where ``X_hat.shape == X.shape`` and
            ``mse.shape == (n,)`` with all values >= 0.

        Raises
        ------
        InferenceError
            If the model has not been built or compiled.
        """
        if self._model is None or self._input_dim is None:
            raise InferenceError(
                "Cannot call reconstruct(): the model is not initialised. "
                "Call build(input_dim) first."
            )
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise InferenceError(
                f"Expected a 2D feature matrix, got shape {X.shape}."
            )
        self._validate_input_dim(X, caller="reconstruct")
        if X.shape[0] == 0:
            raise InferenceError(
                "Cannot reconstruct an empty input batch (0 samples)."
            )

        X_hat = self._model.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_hat), axis=1)
        return X_hat, mse

    def predict(
        self, X: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(binary_predictions, mse_scores)``.

        ``binary_predictions[i] == 1`` iff ``mse_scores[i] >= threshold``.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        threshold : float
            Anomaly decision threshold.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Binary prediction array and corresponding MSE scores.

        Raises
        ------
        InferenceError
            If the model is not built, the batch is empty, or the feature
            dimensionality does not match the training dimensionality.
        """
        if self._model is None or self._input_dim is None:
            raise InferenceError(
                "Cannot call predict(): the model is not initialised. "
                "Call build(input_dim) first."
            )
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise InferenceError(
                f"Expected a 2D feature matrix, got shape {X.shape}."
            )
        self._validate_input_dim(X, caller="predict")
        if X.shape[0] == 0:
            raise InferenceError(
                "Cannot run predict() on an empty input batch (0 samples)."
            )

        _, mse_scores = self.reconstruct(X)
        predictions = (mse_scores >= threshold).astype(int)
        return predictions, mse_scores

    def save_weights(self, path: str) -> None:
        """Persist model weights to *path*.

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"runs/20241215_143022/model_weights.keras"``).

        Raises
        ------
        InferenceError
            If the model has not been built.
        ArtifactWriteError
            If the path is not writable.
        """
        if self._model is None:
            raise InferenceError(
                "Cannot save weights: the model has not been built. "
                "Call build(input_dim) first."
            )
        ok, err = _is_path_writable(os.path.dirname(path) or ".")
        if not ok:
            raise ArtifactWriteError(
                f"The path '{path}' is not writable: {err}"
            )
        if path.endswith(".keras"):
            # Keras 3: a `.keras` file is the full-model (arch + weights) v3
            # format and must be written via Model.save(), not save_weights().
            self._model.save(path, include_optimizer=False)
        else:
            self._model.save_weights(path)
        self._logger.info("Model weights saved to %s", path)

    def load_weights(self, path: str) -> None:
        """Load model weights from *path*.

        Parameters
        ----------
        path : str
            Source file path containing previously saved weights.

        Raises
        ------
        InferenceError
            If the model has not been built (no graph to load weights into).
        """
        if self._model is None:
            raise InferenceError(
                "Cannot load weights: the model has not been built. "
                "Call build(input_dim) first so there is a graph to load into."
            )
        self._model.load_weights(path)
        self._logger.info("Model weights loaded from %s", path)
