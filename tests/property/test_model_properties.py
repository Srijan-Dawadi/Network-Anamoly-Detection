"""
tests/property/test_model_properties.py
========================================
Hypothesis-powered property tests covering the Autoencoder_Model.

      Feature: network-anomaly-autoencoder
      Property 9:  Autoencoder Symmetric Architecture with BatchNorm and Dropout
      Property 10: ConfigurationError for Out-of-Range Parameters
      Property 11: reconstruct(X) Output Shape Invariant
      Property 12: predict(X) Binary Threshold Classification Correctness

Validates Requirements 3.1, 3.2, 3.3, 3.4, 3.6, 4.2, 6.1, 6.2, 6.5.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tensorflow.keras import layers

from network_anomaly_autoencoder.config import ArchitectureConfig, TrainingConfig
from network_anomaly_autoencoder.exceptions import ConfigurationError, InferenceError
from network_anomaly_autoencoder.model import Autoencoder_Model

logger = logging.getLogger("property")

_INPUT_DIM = 16

_ACTIVATIONS = st.sampled_from(["relu", "tanh", "sigmoid"])


def _training_config(**overrides) -> TrainingConfig:
    base = dict(
        learning_rate=0.001,
        batch_size=8,
        max_epochs=2,
        patience=2,
        lr_schedule="none",
        random_seed=0,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _arch_for(layers_list, bottleneck, dropout, activation="relu") -> ArchitectureConfig:
    return ArchitectureConfig(
        encoder_layers=layers_list,
        bottleneck_dim=bottleneck,
        activation=activation,
        dropout_rate=dropout,
    )


def _ordered_layers(model) -> list[tuple[str, object]]:
    """Return (name, layer) for Dense/BN/Dropout layers in forward order."""
    out: list[tuple[str, object]] = []
    for layer in model._model.layers:
        if isinstance(layer, (layers.Dense, layers.BatchNormalization, layers.Dropout)):
            out.append((layer.name, layer))
    return out


# ---------------------------------------------------------------------------
# Property 9: Symmetric architecture with BatchNorm and Dropout
# ---------------------------------------------------------------------------

_valid_layer_sizes = st.lists(
    st.integers(min_value=4, max_value=128), min_size=3, max_size=3
)


@st.composite
def _arch_strategy(draw):
    encoders = draw(_valid_layer_sizes)
    min_enc = min(encoders)
    bottleneck = draw(st.integers(min_value=1, max_value=max(1, min_enc - 1)))
    dropout = draw(st.floats(min_value=0.0, max_value=0.49, allow_nan=False, allow_infinity=False))
    return _arch_for(encoders, bottleneck, dropout, draw(_ACTIVATIONS))


@given(arch=_arch_strategy())
@settings(max_examples=8)
def test_property9_symmetric_batchnorm_dropout(arch):
    # Feature: network-anomaly-autoencoder, Property 9
    model = Autoencoder_Model(arch, _training_config(), logger)
    model.build(_INPUT_DIM)

    encoders = arch.encoder_layers
    expected_dense_units = encoders + [arch.bottleneck_dim] + list(reversed(encoders)) + [_INPUT_DIM]

    dense_units_actual = [
        int(layer.units)
        for layer in model._model.layers
        if isinstance(layer, layers.Dense)
    ]
    # Symmetry: encoder sizes, bottleneck, mirrored decoder sizes, output dim.
    assert dense_units_actual == expected_dense_units

    names = [name for name, _ in _ordered_layers(model)]

    # Every encoder/decoder hidden Dense (enc_dense_*, dec_dense_*) is followed
    # by a BatchNormalization layer.  The bottleneck and final output Dense are
    # intentionally not batchnorm'd (they are not mirrored hidden layers).
    def _hidden_dense_indices():
        return [
            i for i, name in enumerate(names)
            if name.startswith("enc_dense_") or name.startswith("dec_dense_")
        ]

    for i in _hidden_dense_indices():
        nxt = names[i + 1] if i + 1 < len(names) else None
        assert nxt is not None and (
            nxt.startswith("enc_bn_") or nxt.startswith("dec_bn_")
        ), (
            f"Hidden Dense '{names[i]}' not followed by BatchNormalization: {names}"
        )

    if arch.dropout_rate > 0:
        assert any(name.startswith("enc_dropout_") for name in names), (
            "Expected Dropout layers after encoder BN when dropout_rate > 0"
        )
        # Every encoder/decoder BN is followed by its matching Dropout.
        for i, name in enumerate(names):
            if name.startswith("enc_bn_") or name.startswith("dec_bn_"):
                nxt = names[i + 1] if i + 1 < len(names) else None
                assert nxt is not None and nxt.startswith(name.replace("_bn_", "_dropout_")), (
                    f"BatchNormalization '{name}' not followed by Dropout: {names}"
                )
    else:
        assert not any(name.startswith("enc_dropout_") or name.startswith("dec_dropout_") for name in names), (
            "No Dropout layers should appear when dropout_rate == 0"
        )


# ---------------------------------------------------------------------------
# Property 10: ConfigurationError for out-of-range parameters
# ---------------------------------------------------------------------------

_PROBLEM_PARAM = st.sampled_from(
    ["encoder_layer_small", "bottleneck_too_big", "dropout_negative", "dropout_ge_one"]
)


@given(problem=_PROBLEM_PARAM)
@settings(max_examples=20)
def test_property10a_build_raises_configerror_naming_parameter(problem):
    # Feature: network-anomaly-autoencoder, Property 10 (build / architecture)
    if problem == "encoder_layer_small":
        arch = _arch_for([16, 0, 4], 2, 0.0)
        param_name = "encoder_layers"
    elif problem == "bottleneck_too_big":
        arch = _arch_for([16, 8, 4], 16, 0.0)
        param_name = "bottleneck_dim"
    elif problem == "dropout_negative":
        arch = _arch_for([16, 8, 4], 2, -0.5)
        param_name = "dropout_rate"
    else:  # dropout_ge_one
        arch = _arch_for([16, 8, 4], 2, 1.0)
        param_name = "dropout_rate"

    model = Autoencoder_Model(arch, _training_config(), logger)
    try:
        model.build(_INPUT_DIM)
        assert False, f"Expected ConfigurationError for invalid '{param_name}'"
    except ConfigurationError as exc:
        assert param_name in str(exc), (
            f"Error message should name '{param_name}', got: {exc}"
        )


_BAD_TRAIN_PARAM = st.sampled_from(
    ["lr_zero", "lr_negative", "batch_zero", "batch_negative", "epochs_zero", "patience_zero"]
)


@given(problem=_BAD_TRAIN_PARAM)
@settings(max_examples=20)
def test_property10b_fit_raises_configerror_naming_parameter(problem):
    # Feature: network-anomaly-autoencoder, Property 10 (fit / training)
    if problem == "lr_zero":
        train = _training_config(learning_rate=0.0)
        param_name = "learning_rate"
    elif problem == "lr_negative":
        train = _training_config(learning_rate=-0.5)
        param_name = "learning_rate"
    elif problem == "batch_zero":
        train = _training_config(batch_size=0)
        param_name = "batch_size"
    elif problem == "batch_negative":
        train = _training_config(batch_size=-3)
        param_name = "batch_size"
    elif problem == "epochs_zero":
        train = _training_config(max_epochs=0)
        param_name = "max_epochs"
    else:  # patience_zero
        train = _training_config(patience=0)
        param_name = "patience"

    arch = _arch_for([16, 8, 4], 2, 0.0)
    model = Autoencoder_Model(arch, train, logger)
    model.build(_INPUT_DIM)
    X = np.random.default_rng(0).normal(size=(20, _INPUT_DIM)).astype("float32")
    try:
        model.fit(X, X, ".")
        assert False, f"Expected ConfigurationError for invalid '{param_name}'"
    except ConfigurationError as exc:
        assert param_name in str(exc), (
            f"Error message should name '{param_name}', got: {exc}"
        )


# ---------------------------------------------------------------------------
# Property 11: reconstruct(X) output shape invariant
# ---------------------------------------------------------------------------

_N_ROWS = st.integers(min_value=1, max_value=32)


@given(n=_N_ROWS)
@settings(max_examples=10, deadline=None)
def test_property11_reconstruct_shape_invariant(n):
    # Feature: network-anomaly-autoencoder, Property 11
    arch = _arch_for([16, 8, 4], 2, 0.0)
    model = Autoencoder_Model(arch, _training_config(), logger)
    model.build(_INPUT_DIM)
    X = np.random.default_rng(1).normal(size=(n, _INPUT_DIM)).astype("float32")
    X_hat, mse = model.reconstruct(X)
    assert X_hat.shape == (n, _INPUT_DIM)
    assert mse.shape == (n,)
    assert np.all(mse >= 0.0)


# ---------------------------------------------------------------------------
# Property 12: predict(X) binary threshold classification correctness
# ---------------------------------------------------------------------------

_THRESHOLD = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@given(n=_N_ROWS, tau=_THRESHOLD)
@settings(max_examples=10, deadline=None)
def test_property12_predict_classification_correctness(n, tau):
    # Feature: network-anomaly-autoencoder, Property 12
    arch = _arch_for([16, 8, 4], 2, 0.0)
    model = Autoencoder_Model(arch, _training_config(), logger)
    model.build(_INPUT_DIM)
    X = np.random.default_rng(2).normal(size=(n, _INPUT_DIM)).astype("float32")
    predictions, mse_scores = model.predict(X, tau)
    assert predictions.shape == (n,)
    assert mse_scores.shape == (n,)
    expected = (mse_scores >= tau).astype(int)
    assert np.array_equal(predictions, expected)


@given(tau=_THRESHOLD)
@settings(max_examples=10, deadline=None)
def test_property12_dim_mismatch_raises_inference_error(tau):
    # Feature: network-anomaly-autoencoder, Property 12 (dim mismatch)
    arch = _arch_for([16, 8, 4], 2, 0.0)
    model = Autoencoder_Model(arch, _training_config(), logger)
    model.build(_INPUT_DIM)
    X_bad = np.random.default_rng(3).normal(size=(4, _INPUT_DIM + 3)).astype("float32")
    try:
        model.predict(X_bad, tau)
        assert False, "Expected InferenceError for feature-dimension mismatch"
    except InferenceError as exc:
        msg = str(exc)
        assert str(_INPUT_DIM) in msg and str(_INPUT_DIM + 3) in msg
