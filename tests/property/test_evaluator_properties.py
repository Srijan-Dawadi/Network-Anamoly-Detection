"""
tests/property/test_evaluator_properties.py
============================================
Hypothesis-powered property tests covering the Evaluator.

      Feature: network-anomaly-autoencoder
      Property 14: Evaluator Metrics Validity and JSON Completeness

Validates Requirements 7.1, 7.2, 7.6.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from network_anomaly_autoencoder.evaluator import Evaluator

logger = logging.getLogger("property")

_REQUIRED_METRIC_KEYS = (
    "precision_normal",
    "recall_normal",
    "f1_normal",
    "precision_anomalous",
    "recall_anomalous",
    "f1_anomalous",
    "accuracy",
    "auc_roc",
)


@pytest.fixture(scope="session")
def _tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("evaluator_props")


def _newdir(_tmp) -> str:
    d = _tmp / uuid.uuid4().hex
    d.mkdir(parents=True)
    return str(d)


@st.composite
def _label_and_scores(draw):
    n_normal = draw(st.integers(min_value=1, max_value=30))
    n_anom = draw(st.integers(min_value=1, max_value=30))
    y_true = np.array([0] * n_normal + [1] * n_anom)
    normal_scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_normal,
            max_size=n_normal,
        )
    )
    anom_scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n_anom,
            max_size=n_anom,
        )
    )
    scores = np.array(normal_scores + anom_scores, dtype=float)
    # Shuffle so the model ordering is not trivially class-separated.
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    perm = rng.permutation(len(y_true))
    return y_true[perm], scores[perm]


@given(pair=_label_and_scores())
@settings(max_examples=30, deadline=None)
def test_property14_metrics_valid_and_json_complete(_tmp, pair):
    # Feature: network-anomaly-autoencoder, Property 14
    y_true, mse_scores = pair
    y_pred = (mse_scores >= np.median(mse_scores)).astype(int)

    d = _newdir(_tmp)
    evaluator = Evaluator(logger)
    metrics = evaluator.evaluate(y_true, y_pred, mse_scores, float(np.median(mse_scores)), d)

    assert set(_REQUIRED_METRIC_KEYS) <= set(metrics.keys())
    for key in _REQUIRED_METRIC_KEYS:
        value = metrics[key]
        assert 0.0 <= value <= 1.0, f"Metric '{key}' out of range: {value}"

    with open(os.path.join(d, "metrics.json"), "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    assert set(_REQUIRED_METRIC_KEYS) <= set(saved.keys())

    # The three required plots must be produced.
    for plot in ("confusion_matrix.png", "roc_curve.png", "re_histogram.png"):
        assert os.path.isfile(os.path.join(d, plot)), f"Missing plot: {plot}"


@given(n=st.integers(min_value=1, max_value=30))
@settings(max_examples=15, deadline=None)
def test_property14_perfect_separation_auc_one(_tmp, n):
    # Feature: network-anomaly-autoencoder, Property 14 (perfect separation)
    y_true = np.array([0] * n + [1] * n)
    # All attack scores strictly above all normal scores => perfect separation.
    mse_scores = np.array(list(np.linspace(0.0, 1.0, n)) + list(np.linspace(2.0, 3.0, n)))
    y_pred = (mse_scores >= 1.5).astype(int)

    d = _newdir(_tmp)
    evaluator = Evaluator(logger)
    metrics = evaluator.evaluate(y_true, y_pred, mse_scores, 1.5, d)
    assert metrics["auc_roc"] == pytest.approx(1.0)
