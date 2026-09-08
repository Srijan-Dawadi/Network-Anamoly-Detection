"""
tests/unit/test_model_build.py
==============================
Unit tests for ``Autoencoder_Model.build``.

Validates Requirements 3.2, 3.3, 3.4, 3.5.

``model.py`` imports TensorFlow at module level, so all tests in this file
are skipped when TF is not available (via the module-level
``pytest.importorskip`` guard).
"""

from __future__ import annotations

import logging

import pytest

# model.py imports TensorFlow at the top level, so we must skip the entire
# module if TF is absent – not just individual test methods.
tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed")

from network_anomaly_autoencoder.config import ArchitectureConfig, TrainingConfig  # noqa: E402
from network_anomaly_autoencoder.exceptions import ConfigurationError  # noqa: E402
from network_anomaly_autoencoder.model import Autoencoder_Model  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arch(
    encoder_layers: list[int] | None = None,
    bottleneck_dim: int = 16,
    activation: str = "relu",
    dropout_rate: float = 0.0,
) -> ArchitectureConfig:
    """Return an ``ArchitectureConfig`` with sensible defaults."""
    return ArchitectureConfig(
        encoder_layers=encoder_layers if encoder_layers is not None else [64, 32],
        bottleneck_dim=bottleneck_dim,
        activation=activation,
        dropout_rate=dropout_rate,
    )


def _make_train(
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
    patience: int = 3,
    lr_schedule: str = "none",
    random_seed: int = 42,
) -> TrainingConfig:
    """Return a ``TrainingConfig`` with sensible defaults."""
    return TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr_schedule=lr_schedule,
        random_seed=random_seed,
    )


def _make_model(
    arch: ArchitectureConfig | None = None,
    train: TrainingConfig | None = None,
) -> Autoencoder_Model:
    """Construct an ``Autoencoder_Model`` with default configs."""
    return Autoencoder_Model(
        arch_config=arch or _make_arch(),
        train_config=train or _make_train(),
        logger=logging.getLogger("test"),
    )


# ---------------------------------------------------------------------------
# ConfigurationError tests – validation fires before graph construction,
# but the class still requires TF to be importable (guarded above).
# ---------------------------------------------------------------------------

class TestBuildValidation:
    """build() must raise ConfigurationError for each invalid arch parameter (Req 3.3)."""

    def test_encoder_layer_size_less_than_one(self):
        """encoder_layers containing 0 raises ConfigurationError."""
        arch = _make_arch(encoder_layers=[0, 32], bottleneck_dim=8)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="encoder_layers"):
            model.build(input_dim=10)

    def test_bottleneck_dim_equal_to_min_encoder(self):
        """bottleneck_dim == min(encoder_layers) raises ConfigurationError."""
        # encoder=[64, 32], bottleneck=32  →  32 >= 32  → invalid
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=32)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="bottleneck_dim"):
            model.build(input_dim=10)

    def test_bottleneck_dim_greater_than_min_encoder(self):
        """bottleneck_dim > min(encoder_layers) raises ConfigurationError."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=40)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="bottleneck_dim"):
            model.build(input_dim=10)

    def test_bottleneck_dim_less_than_one(self):
        """bottleneck_dim == 0 raises ConfigurationError."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=0)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="bottleneck_dim"):
            model.build(input_dim=10)

    def test_dropout_rate_equal_to_one(self):
        """dropout_rate == 1.0 raises ConfigurationError."""
        arch = _make_arch(dropout_rate=1.0)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="dropout_rate"):
            model.build(input_dim=10)

    def test_dropout_rate_greater_than_one(self):
        """dropout_rate > 1.0 raises ConfigurationError."""
        arch = _make_arch(dropout_rate=1.5)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="dropout_rate"):
            model.build(input_dim=10)

    def test_dropout_rate_negative(self):
        """dropout_rate == -0.1 raises ConfigurationError."""
        arch = _make_arch(dropout_rate=-0.1)
        model = _make_model(arch=arch)
        with pytest.raises(ConfigurationError, match="dropout_rate"):
            model.build(input_dim=10)


# ---------------------------------------------------------------------------
# Functional tests (require TF to actually build the Keras graph)
# ---------------------------------------------------------------------------

class TestBuildFunctional:
    """Functional behaviour of a successfully built model."""

    def test_build_succeeds_with_valid_params(self):
        """build() must not raise for valid architecture parameters (Req 3.2)."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=16, dropout_rate=0.2)
        model = _make_model(arch=arch)
        # Should complete without raising
        model.build(input_dim=10)

    def test_build_sets_input_dim(self):
        """build() must store the passed input_dim on the model (Req 3.4)."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=16)
        model = _make_model(arch=arch)
        model.build(input_dim=41)
        assert model._input_dim == 41

    def test_compiled_loss_is_mse(self):
        """The compiled model must use MSE as the loss function (Req 3.5)."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=16)
        model = _make_model(arch=arch)
        model.build(input_dim=10)
        assert model._model is not None
        assert model._model.loss == "mse"

    def test_no_dropout_layers_when_rate_is_zero(self):
        """When dropout_rate == 0, the built model must contain no Dropout layers (Req 3.3)."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=16, dropout_rate=0.0)
        model = _make_model(arch=arch)
        model.build(input_dim=10)

        dropout_layers = [
            layer
            for layer in model._model.layers
            if isinstance(layer, tf.keras.layers.Dropout)
        ]
        assert len(dropout_layers) == 0, (
            f"Expected no Dropout layers when dropout_rate=0.0, "
            f"but found: {[layer.name for layer in dropout_layers]}"
        )

    def test_dropout_layers_present_when_rate_is_nonzero(self):
        """When dropout_rate > 0, the built model must contain Dropout layers (Req 3.3)."""
        arch = _make_arch(encoder_layers=[64, 32], bottleneck_dim=16, dropout_rate=0.3)
        model = _make_model(arch=arch)
        model.build(input_dim=10)

        dropout_layers = [
            layer
            for layer in model._model.layers
            if isinstance(layer, tf.keras.layers.Dropout)
        ]
        assert len(dropout_layers) > 0, (
            "Expected Dropout layers when dropout_rate=0.3, but found none."
        )

    def test_build_with_single_encoder_layer(self):
        """build() must succeed with a single encoder layer (Req 3.2)."""
        arch = _make_arch(encoder_layers=[64], bottleneck_dim=16, dropout_rate=0.0)
        model = _make_model(arch=arch)
        model.build(input_dim=20)
        assert model._model is not None

    def test_build_with_deep_encoder(self):
        """build() must succeed with a three-layer encoder (Req 3.2)."""
        arch = _make_arch(encoder_layers=[128, 64, 32], bottleneck_dim=8, dropout_rate=0.0)
        model = _make_model(arch=arch)
        model.build(input_dim=30)
        assert model._model is not None
