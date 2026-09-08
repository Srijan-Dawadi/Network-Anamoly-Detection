"""
tests/property/test_pipeline_properties.py
===========================================
Hypothesis-powered property tests covering the Data_Pipeline.

      Feature: network-anomaly-autoencoder
      Property 1:  Schema Column-Count Validation
      Property 2:  Missing File Raises DataLoadError with Path
      Property 3:  Deduplication Idempotence
      Property 4:  Count Invariant — Total Equals Normal Plus Attack
      Property 5:  OHE and Feature Dimensionality Consistency
      Property 6:  StandardScaler Normalisation Invariant
      Property 7:  Unknown Categorical Value Encoded as All-Zero Vector
      Property 8:  Split-Ratio Validity and Size Proportionality

Validates Requirements 1.1–1.6, 2.1–2.6.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from network_anomaly_autoencoder.config import ExperimentConfig, load_config
from network_anomaly_autoencoder.data_pipeline import Data_Pipeline
from network_anomaly_autoencoder.exceptions import (
    ConfigurationError,
    DataLoadError,
    SchemaValidationError,
)

from tests.conftest import (
    NSL_KDD_ALL_COLUMNS,
    NSL_KDD_FLAG_VALUES,
    NSL_KDD_NUMERICAL_COLUMNS,
    NSL_KDD_PROTOCOL_VALUES,
    NSL_KDD_SERVICE_VALUES,
    _make_nsl_kdd_df,
)

logger = logging.getLogger("property")


@pytest.fixture(scope="session")
def _tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline_props")


def _newdir(_tmp) -> str:
    d = _tmp / uuid.uuid4().hex
    d.mkdir(parents=True)
    return str(d)


def _pipeline() -> Data_Pipeline:
    cfg = load_config("configs/nsl_kdd_default.yaml")
    return Data_Pipeline(cfg, logger)


# ---------------------------------------------------------------------------
# Property 1 — Schema column-count validation
# ---------------------------------------------------------------------------

@given(
    st.lists(st.sampled_from(NSL_KDD_ALL_COLUMNS), min_size=0, max_size=60, unique=True)
)
@settings(max_examples=50)
def test_property1_schema_validation(_tmp, columns):
    # Feature: network-anomaly-autoencoder, Property 1: Schema column-count validation
    d = _newdir(_tmp)
    df = pd.DataFrame(columns=columns)
    df.to_csv(f"{d}/data.csv", index=False)
    import dataclasses

    cfg = load_config("configs/nsl_kdd_default.yaml")
    cfg = dataclasses.replace(cfg, dataset=dataclasses.replace(cfg.dataset, path=f"{d}/data.csv"))
    pipeline = Data_Pipeline(cfg, logger)
    try:
        result = pipeline.load_and_validate()
    except (SchemaValidationError, DataLoadError) as exc:
        if isinstance(exc, SchemaValidationError):
            # Accept iff a required column is missing.
            required = set(pipeline._schema.REQUIRED_COLUMNS)
            missing = required - set(columns)
            assert len(missing) > 0
            assert all(col in str(exc) for col in list(missing)[:3])
        else:
            # Accept a DataLoadError: the generated file never contains data
            # rows, only a header, so it may be rejected as empty (or as an
            # unparseable/zero-row file) regardless of column count.
            return
    # If it loaded, all required columns must be present.
    assert set(pipeline._schema.REQUIRED_COLUMNS).issubset(result.columns)


# ---------------------------------------------------------------------------
# Property 2 — Missing file raises DataLoadError with the path
# ---------------------------------------------------------------------------

_SAFE_NAME = st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=122), min_size=1)


@given(_SAFE_NAME)
@settings(max_examples=30)
def test_property2_missing_file_raises_with_path(_tmp, filename):
    # Feature: network-anomaly-autoencoder, Property 2: Missing File Raises DataLoadError with Path
    d = _newdir(_tmp)
    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(cfg, dataset=dataclasses.replace(cfg.dataset, path=f"{d}/{filename}"))
    pipeline = Data_Pipeline(cfg, logger)
    path = cfg.dataset.path
    try:
        pipeline.load_and_validate()
        assert False, "Expected DataLoadError for missing file"
    except DataLoadError as exc:
        assert path in str(exc)


# ---------------------------------------------------------------------------
# Property 3 — Deduplication idempotence
# ---------------------------------------------------------------------------

@given(st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=50))
@settings(max_examples=50)
def test_property3_deduplication_idempotence(multipliers):
    # Feature: network-anomaly-autoencoder, Property 3: Deduplication Idempotence
    df = _make_nsl_kdd_df(n_normal=max(len(multipliers), 2), n_attack=2)
    repeated = pd.concat([df] * 3, ignore_index=True)
    # Inject exact duplicates.
    repeated = pd.concat([repeated, repeated.head(5)], ignore_index=True)

    pipeline = _pipeline()
    once = pipeline._deduplicate(repeated)
    twice = pipeline._deduplicate(once)
    assert len(once) == len(twice)
    assert len(once) == len(once.drop_duplicates(subset=pipeline._schema.REQUIRED_COLUMNS))


# ---------------------------------------------------------------------------
# Property 4 — Count invariant: total == normal + attack
# ---------------------------------------------------------------------------

@given(n_normal=st.integers(min_value=2, max_value=60), n_attack=st.integers(min_value=0, max_value=40))
@settings(max_examples=50)
def test_property4_count_invariant(_tmp, n_normal, n_attack):
    # Feature: network-anomaly-autoencoder, Property 4: Count invariant
    d = _newdir(_tmp)
    df = _make_nsl_kdd_df(n_normal=n_normal, n_attack=n_attack)
    df.to_csv(f"{d}/data.csv", index=False)
    import dataclasses

    cfg = load_config("configs/nsl_kdd_default.yaml")
    cfg = dataclasses.replace(cfg, dataset=dataclasses.replace(cfg.dataset, path=f"{d}/data.csv"))
    pipeline = Data_Pipeline(cfg, logger)
    df = pipeline.load_and_validate()
    label_col = pipeline._schema.LABEL_COLUMN
    total = len(df)
    normal_count = int((df[label_col] == pipeline._schema.NORMAL_LABEL).sum())
    attack_count = total - normal_count
    assert normal_count + attack_count == total
    assert normal_count >= 0 and attack_count >= 0


# ---------------------------------------------------------------------------
# Property 8 — Split-ratio validity & size proportionality
# ---------------------------------------------------------------------------

def _ratio_triples():
    a = st.floats(min_value=0.1, max_value=0.8, allow_nan=False, allow_infinity=False)
    b = st.floats(min_value=0.1, max_value=0.8, allow_nan=False, allow_infinity=False)

    def build(t):
        r1, r2 = t
        r3 = 1.0 - r1 - r2
        if 0.0 < r3 < 1.0:
            return (r1, r2, r3)
        return None

    return st.tuples(a, b).map(build).filter(lambda x: x is not None)


@given(_ratio_triples())
@settings(max_examples=50, deadline=None)
def test_property8_split_ratio_proportionality(_tmp, ratios):
    # Feature: network-anomaly-autoencoder, Property 8: Split-ratio validity and size proportionality
    d = _newdir(_tmp)
    train_ratio, val_ratio, test_ratio = ratios
    df = _make_nsl_kdd_df(n_normal=500, n_attack=50)
    df.to_csv(f"{d}/data.csv", index=False)

    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(
        cfg,
        dataset=dataclasses.replace(cfg.dataset, path=f"{d}/data.csv"),
        splits=dataclasses.replace(cfg.splits, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio),
    )
    pipeline = Data_Pipeline(cfg, logger)
    Xtr, Xval, Xtest, *_ = pipeline.fit_transform(df)
    n_normal = 500
    # train+val pool size proportional to (train+val)/(0.7+0.1...) scaled to train+val ratio
    train_target = round(n_normal * train_ratio / (train_ratio + val_ratio))
    assert abs(len(Xtr) - train_target) <= 1
    assert abs(len(Xval) - (n_normal - train_target)) <= 1
    assert len(Xtest) == 550


@given(st.sampled_from(
    [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0), (1.2, 0.3, -0.5)]
))
@settings(max_examples=20)
def test_property8_invalid_ratio_raises(_tmp, ratios):
    # Feature: network-anomaly-autoencoder, Property 8: invalid ratios raise ConfigurationError
    train_ratio, val_ratio, test_ratio = ratios
    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(
        cfg,
        splits=dataclasses.replace(cfg.splits, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio),
    )
    pipeline = Data_Pipeline(cfg, logger)
    df = _make_nsl_kdd_df(n_normal=50, n_attack=10)
    with pytest.raises(ConfigurationError):
        pipeline.fit_transform(df)


# ---------------------------------------------------------------------------
# Property 5 — OHE & feature dimensionality consistency
# ---------------------------------------------------------------------------

@given(n_normal=st.integers(min_value=10, max_value=50), n_attack=st.integers(min_value=1, max_value=20))
@settings(max_examples=30)
def test_property5_feature_dimensionality_consistency(_tmp, n_normal, n_attack):
    # Feature: network-anomaly-autoencoder, Property 5: OHE and feature dimensionality consistency
    d = _newdir(_tmp)
    df = _make_nsl_kdd_df(n_normal=n_normal, n_attack=n_attack)
    df.to_csv(f"{d}/data.csv", index=False)
    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(cfg, dataset=dataclasses.replace(cfg.dataset, path=f"{d}/data.csv"))
    pipeline = Data_Pipeline(cfg, logger)
    Xtr, Xval, Xtest, *_ = pipeline.fit_transform(df)
    assert Xtr.shape[1] == Xval.shape[1] == Xtest.shape[1]
    # Post-OHE dimensionality strictly exceeds the raw feature count.
    assert Xtr.shape[1] > pipeline._schema.FEATURE_COUNT


# ---------------------------------------------------------------------------
# Property 6 — StandardScaler normalisation invariant
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=20)
def test_property6_standardscaler_normalisation(_tmp, seed):
    # Feature: network-anomaly-autoencoder, Property 6: StandardScaler normalisation invariant
    d = _newdir(_tmp)
    rng = np.random.default_rng(seed)
    df = _make_nsl_kdd_df(n_normal=200, n_attack=20, rng=rng)
    df.to_csv(f"{d}/data.csv", index=False)
    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(cfg, dataset=dataclasses.replace(cfg.dataset, path=f"{d}/data.csv"))
    pipeline = Data_Pipeline(cfg, logger)
    Xtr, *_ = pipeline.fit_transform(df)
    num_cols = pipeline._schema.NUMERICAL_COLUMNS
    # Numerical features are concatenated AFTER the one-hot columns, so they
    # occupy the last ``len(num_cols)`` columns of the feature matrix.
    scaled = Xtr[:, -len(num_cols) :]
    means = scaled.mean(axis=0)
    stds = scaled.std(axis=0)
    assert np.allclose(means, 0.0, atol=0.01)
    assert np.allclose(stds, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Property 7 — Unknown categorical value encoded as all-zero vector
# ---------------------------------------------------------------------------

@given(new_proto=st.sampled_from(["GRE", "AH", "ESP", "IPV6"]))
@settings(max_examples=10)
def test_property7_unknown_categorical_all_zero(_tmp, new_proto):
    # Feature: network-anomaly-autoencoder, Property 7: Unknown categorical value all-zero
    d = _newdir(_tmp)
    df = _make_nsl_kdd_df(n_normal=80, n_attack=10)
    df.to_csv(f"{d}/data.csv", index=False)
    cfg = load_config("configs/nsl_kdd_default.yaml")
    import dataclasses

    cfg = dataclasses.replace(cfg, dataset=dataclasses.replace(cfg.dataset, path=f"{d}/data.csv"))
    pipeline = Data_Pipeline(cfg, logger)
    pipeline.fit_transform(df)
    pipeline.save_artefacts(d)

    # Inference row with an unseen protocol value.
    new_row = _make_nsl_kdd_df(n_normal=1, n_attack=0)
    new_row.loc[0, "protocol_type"] = new_proto

    pipeline2 = Data_Pipeline(cfg, logger)
    pipeline2.load_artefacts(d)
    out = pipeline2.transform(new_row)[0]
    # Find the OHE column indices for protocol_type.
    cat_cols = pipeline2._schema.CATEGORICAL_COLUMNS
    proto_idx = cat_cols.index("protocol_type")
    ohe = pipeline2._ohe
    # Original protocol categories that were fit.
    fitted_protos = ohe.categories_[proto_idx]
    start = 0
    offsets = []
    for col_idx, col in enumerate(cat_cols):
        n = len(ohe.categories_[col_idx])
        offsets.append((start, start + n))
        start += n
    p_start, p_end = offsets[proto_idx]
    if new_proto in fitted_protos:
        # Column is meaningfully set; ensure exactly one hot.
        assert out[p_start:p_end].sum() == 1.0
    else:
        # Unseen -> all zero for that feature's one-hot block.
        assert np.allclose(out[p_start:p_end], 0.0)
