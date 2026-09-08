"""
tests/property/test_threshold_properties.py
============================================
Hypothesis-powered property tests covering the Threshold_Estimator.

      Feature: network-anomaly-autoencoder
      Property 13: Threshold Round-Trip — Compute, Save, Load

Validates Requirements 5.1, 5.2, 5.3, 5.4.
"""

from __future__ import annotations

import json
import logging
import uuid

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from network_anomaly_autoencoder.config import ThresholdConfig
from network_anomaly_autoencoder.exceptions import ArtifactLoadError
from network_anomaly_autoencoder.threshold import Threshold_Estimator

logger = logging.getLogger("property")

_REQUIRED_KEYS = ("threshold", "percentile", "mse_mean", "mse_std")


@pytest.fixture(scope="session")
def _tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("threshold_props")


def _newdir(_tmp) -> str:
    d = _tmp / uuid.uuid4().hex
    d.mkdir(parents=True)
    return str(d)


_MSE_VEC = st.lists(
    st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=2,
)
_PERCENTILE = st.integers(min_value=1, max_value=99)


@given(mse=_MSE_VEC, percentile=_PERCENTILE)
@settings(max_examples=50)
def test_property13_round_trip(_tmp, mse, percentile):
    # Feature: network-anomaly-autoencoder, Property 13: Threshold round-trip
    config = ThresholdConfig(percentile=percentile)
    mse_arr = np.asarray(mse, dtype=float)

    estimator = Threshold_Estimator(config, logger)
    threshold = estimator.fit(mse_arr)
    expected = float(np.percentile(mse_arr, percentile))
    assert threshold == pytest.approx(expected, rel=1e-6, abs=1e-6)

    d = _newdir(_tmp)
    estimator.save(d)

    # JSON file contains all four required fields with matching values.
    import os

    with open(os.path.join(d, "threshold.json"), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert set(_REQUIRED_KEYS) <= set(payload.keys())
    assert payload["threshold"] == pytest.approx(threshold, rel=1e-6, abs=1e-6)
    assert payload["percentile"] == percentile
    assert payload["mse_mean"] == pytest.approx(float(np.mean(mse_arr)), rel=1e-6, abs=1e-6)
    assert payload["mse_std"] == pytest.approx(float(np.std(mse_arr)), rel=1e-6, abs=1e-6)

    # Load returns the same threshold value.
    estimator2 = Threshold_Estimator(config, logger)
    loaded = estimator2.load(d)
    assert loaded == pytest.approx(threshold, rel=1e-6, abs=1e-6)


@given(missing_key=st.sampled_from(_REQUIRED_KEYS))
@settings(max_examples=20)
def test_property13_missing_field_raises_artifact_load_error(_tmp, missing_key):
    # Feature: network-anomaly-autoencoder, Property 13: invalid artefact
    d = _newdir(_tmp)
    import os

    payload = {
        "threshold": 0.5,
        "percentile": 95,
        "mse_mean": 0.1,
        "mse_std": 0.2,
    }
    del payload[missing_key]
    with open(os.path.join(d, "threshold.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    config = ThresholdConfig(percentile=95)
    estimator = Threshold_Estimator(config, logger)
    try:
        estimator.load(d)
        assert False, f"Expected ArtifactLoadError for missing field '{missing_key}'"
    except ArtifactLoadError as exc:
        assert missing_key in str(exc)


@given(junk=st.text(min_size=1, max_size=50))
@settings(max_examples=10)
def test_property13_invalid_json_raises_artifact_load_error(_tmp, junk):
    # Feature: network-anomaly-autoencoder, Property 13: invalid JSON
    import os

    d = _newdir(_tmp)
    with open(os.path.join(d, "threshold.json"), "w", encoding="utf-8") as fh:
        fh.write(junk)

    config = ThresholdConfig(percentile=95)
    estimator = Threshold_Estimator(config, logger)
    try:
        estimator.load(d)
        assert False, "Expected ArtifactLoadError for invalid JSON"
    except ArtifactLoadError:
        pass
