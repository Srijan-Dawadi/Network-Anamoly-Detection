"""
tests/unit/test_model.py
========================
Unit tests for ``Autoencoder_Model.fit``, ``reconstruct``, ``predict``,
``save_weights``, and ``load_weights``.

Validates Requirements 3.6, 4.3, 4.6, 4.7, 6.3, 6.5, 6.6.

Skipped when TensorFlow is not installed.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed")

from network_anomaly_autoencoder.config import ArchitectureConfig, TrainingConfig  # noqa: E402
from network_anomaly_autoencoder.exceptions import (  # noqa: E402
    ArtifactWriteError,
    DataLoadError,
    InferenceError,
)
from network_anomaly_autoencoder.model import Autoencoder_Model  # noqa: E402


def _make_arch(
    encoder_layers: list[int] | None = None,
    bottleneck_dim: int = 8,
    dropout_rate: float = 0.0,
) -> ArchitectureConfig:
    return ArchitectureConfig(
        encoder_layers=encoder_layers if encoder_layers is not None else [32, 16],
        bottleneck_dim=bottleneck_dim,
        activation="relu",
        dropout_rate=dropout_rate,
    )


def _make_train(
    max_epochs: int = 5,
    patience: int = 2,
    random_seed: int = 42,
    **kwargs,
) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=1e-3,
        batch_size=16,
        max_epochs=max_epochs,
        patience=patience,
        lr_schedule="none",
        random_seed=random_seed,
        **kwargs,
    )


def _make_model(
    arch: ArchitectureConfig | None = None,
    train: TrainingConfig | None = None,
) -> Autoencoder_Model:
    return Autoencoder_Model(
        arch_config=arch or _make_arch(),
        train_config=train or _make_train(),
        logger=logging.getLogger("test"),
    )


def _rand(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=shape).astype("float32")


class TestFit:
    """fit() training behaviour and error handling (Req 4)."""

    def test_fit_raises_inference_error_before_build(self, tmp_path):
        model = _make_model()
        with pytest.raises(InferenceError, match="not been built"):
            model.fit(_rand((30, 10)), _rand((10, 10)), str(tmp_path))

    def test_fit_raises_dataloaderror_on_empty_train(self, tmp_path):
        model = _make_model()
        model.build(input_dim=10)
        with pytest.raises(DataLoadError, match="zero normal-traffic"):
            model.fit(np.empty((0, 10)), _rand((10, 10)), str(tmp_path))

    def test_fit_raises_artifactwriteerror_on_unwritable_path(self, tmp_path):
        model = _make_model()
        model.build(input_dim=10)
        # A regular file cannot be used as an artefact directory.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        unwritable = str(blocker / "subdir")
        with pytest.raises(ArtifactWriteError):
            model.fit(_rand((30, 10)), _rand((10, 10)), unwritable)

    def test_fit_trains_and_returns_history(self, tmp_path):
        model = _make_model()
        model.build(input_dim=10)
        history = model.fit(_rand((40, 10)), _rand((10, 10)), str(tmp_path))
        assert history is not None
        assert "loss" in history.history
        assert "val_loss" in history.history

    def test_fit_saves_weights_file(self, tmp_path):
        model = _make_model()
        model.build(input_dim=10)
        model.fit(_rand((40, 10)), _rand((10, 10)), str(tmp_path))
        assert os.path.isfile(os.path.join(str(tmp_path), "model_weights.keras"))

    def test_fit_uses_only_normal_monitors_val(self, tmp_path):
        # Ensures validation_data=X_val is wired (X_val acts as val set).
        model = _make_model()
        model.build(input_dim=10)
        history = model.fit(_rand((40, 10)), _rand((10, 10)), str(tmp_path))
        assert len(history.history["val_loss"]) == len(history.history["loss"])


class TestReconstruct:
    """reconstruct() output-shape and state invariants (Req 3.6)."""

    def test_reconstruct_raises_inference_error_before_build(self):
        model = _make_model()
        with pytest.raises(InferenceError, match="not initialised"):
            model.reconstruct(_rand((5, 10)))

    def test_reconstruct_output_shapes(self):
        model = _make_model()
        model.build(input_dim=12)
        X = _rand((7, 12))
        X_hat, mse = model.reconstruct(X)
        assert X_hat.shape == X.shape
        assert mse.shape == (7,)
        assert (mse >= 0).all()

    def test_reconstruct_empty_batch_raises(self):
        model = _make_model()
        model.build(input_dim=12)
        with pytest.raises(InferenceError, match="empty"):
            model.reconstruct(np.empty((0, 12), dtype="float32"))

    def test_reconstruct_dimension_mismatch_raises(self):
        model = _make_model()
        model.build(input_dim=12)
        with pytest.raises(InferenceError, match="expected 12 features but got 5"):
            model.reconstruct(_rand((3, 5)))


class TestPredict:
    """predict() binary threshold classification (Req 6)."""

    def test_predict_raises_inference_error_before_build(self):
        model = _make_model()
        with pytest.raises(InferenceError):
            model.predict(_rand((5, 10)), 0.5)

    def test_predict_binary_threshold_correctness(self):
        model = _make_model()
        model.build(input_dim=8)
        X = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3
            + [[100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]] * 3,
            dtype="float32",
        )
        _, mse = model.reconstruct(X)
        threshold = np.percentile(mse, 95)
        pred, mse_out = model.predict(X, threshold)
        expected = (mse >= threshold).astype(int)
        assert np.array_equal(pred, expected)
        assert np.array_equal(mse_out, mse)

    def test_predict_empty_batch_raises(self):
        model = _make_model()
        model.build(input_dim=8)
        with pytest.raises(InferenceError, match="empty"):
            model.predict(np.empty((0, 8), dtype="float32"), 0.5)

    def test_predict_dimension_mismatch_raises(self):
        model = _make_model()
        model.build(input_dim=8)
        with pytest.raises(InferenceError, match="expected 8 features but got 3"):
            model.predict(_rand((4, 3)), 0.5)


class TestWeightIO:
    """save_weights / load_weights round-trip (Req 4.4, 9.3)."""

    def test_save_weights_raises_before_build(self, tmp_path):
        model = _make_model()
        with pytest.raises(InferenceError, match="not been built"):
            model.save_weights(str(tmp_path / "model_weights.keras"))

    def test_load_weights_raises_before_build(self, tmp_path):
        model = _make_model()
        with pytest.raises(InferenceError, match="not been built"):
            model.load_weights(str(tmp_path / "model_weights.keras"))

    def test_save_weights_unwritable_raises(self, tmp_path):
        model = _make_model()
        model.build(input_dim=8)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        unwritable = str(blocker / "w.keras")
        with pytest.raises(ArtifactWriteError):
            model.save_weights(unwritable)

    def test_weight_round_trip(self, tmp_path):
        arch = _make_arch()
        tr = _make_train(max_epochs=2)
        src = Autoencoder_Model(arch, tr, logging.getLogger("src"))
        src.build(input_dim=8)
        path = os.path.join(str(tmp_path), "model_weights.keras")
        src.save_weights(path)
        assert os.path.isfile(path)

        dst = Autoencoder_Model(arch, tr, logging.getLogger("dst"))
        dst.build(input_dim=8)
        dst.load_weights(path)
        for a, b in zip(src._model.get_weights(), dst._model.get_weights()):
            assert np.allclose(a.astype("float32"), b.astype("float32"))


class TestSeedReproducibility:
    """Two runs with identical config produce identical weights (Req 4.5)."""

    def test_same_seed_same_weights(self, tmp_path):
        arch = _make_arch()
        tr1 = _make_train(random_seed=7, max_epochs=2)
        tr2 = _make_train(random_seed=7, max_epochs=2)

        m1 = Autoencoder_Model(arch, tr1, logging.getLogger("a"))
        m1.build(input_dim=10)
        m1.fit(_rand((40, 10), seed=1), _rand((10, 10), seed=2), str(tmp_path))

        m2 = Autoencoder_Model(arch, tr2, logging.getLogger("b"))
        m2.build(input_dim=10)
        m2.fit(_rand((40, 10), seed=1), _rand((10, 10), seed=2), str(tmp_path))

        for a, b in zip(m1._model.get_weights(), m2._model.get_weights()):
            assert np.allclose(a.astype("float32"), b.astype("float32"))
