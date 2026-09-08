"""
tests/unit/test_schemas.py
==========================
Unit tests for dataset schema modules.

Validates: Requirements 1.1

Tests cover all three schema modules (nsl_kdd, cicids2017, unsw_nb15) and
assert structural correctness of their exported constants.
"""

import pytest

import network_anomaly_autoencoder.schemas.nsl_kdd as nsl_kdd
import network_anomaly_autoencoder.schemas.cicids2017 as cicids2017
import network_anomaly_autoencoder.schemas.unsw_nb15 as unsw_nb15

# ---------------------------------------------------------------------------
# Parametrize data — (module, expected_feature_count, normal_label)
# ---------------------------------------------------------------------------
SCHEMAS = [
    pytest.param(nsl_kdd,    41, "normal", id="nsl_kdd"),
    pytest.param(cicids2017, 80, "BENIGN", id="cicids2017"),
    pytest.param(unsw_nb15,  49, 0,        id="unsw_nb15"),
]


# ---------------------------------------------------------------------------
# 1. FEATURE_COUNT equals the expected value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, expected_count, _normal", SCHEMAS)
def test_feature_count(schema, expected_count, _normal):
    assert schema.FEATURE_COUNT == expected_count


# ---------------------------------------------------------------------------
# 2. CATEGORICAL_COLUMNS is a list (may be empty)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_categorical_columns_is_list(schema, _count, _normal):
    assert isinstance(schema.CATEGORICAL_COLUMNS, list)


# ---------------------------------------------------------------------------
# 3. NUMERICAL_COLUMNS is a non-empty list
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_numerical_columns_is_nonempty_list(schema, _count, _normal):
    assert isinstance(schema.NUMERICAL_COLUMNS, list)
    assert len(schema.NUMERICAL_COLUMNS) > 0


# ---------------------------------------------------------------------------
# 4. REQUIRED_COLUMNS == CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [LABEL_COLUMN]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_required_columns_composition(schema, _count, _normal):
    expected = schema.CATEGORICAL_COLUMNS + schema.NUMERICAL_COLUMNS + [schema.LABEL_COLUMN]
    assert schema.REQUIRED_COLUMNS == expected


# ---------------------------------------------------------------------------
# 5. No duplicates in any column list
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_no_duplicates_in_categorical_columns(schema, _count, _normal):
    assert len(schema.CATEGORICAL_COLUMNS) == len(set(schema.CATEGORICAL_COLUMNS))


@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_no_duplicates_in_numerical_columns(schema, _count, _normal):
    assert len(schema.NUMERICAL_COLUMNS) == len(set(schema.NUMERICAL_COLUMNS))


@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_no_duplicates_in_required_columns(schema, _count, _normal):
    assert len(schema.REQUIRED_COLUMNS) == len(set(schema.REQUIRED_COLUMNS))


# ---------------------------------------------------------------------------
# 6. LABEL_COLUMN is not in CATEGORICAL_COLUMNS or NUMERICAL_COLUMNS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_label_column_not_in_categorical_columns(schema, _count, _normal):
    assert schema.LABEL_COLUMN not in schema.CATEGORICAL_COLUMNS


@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_label_column_not_in_numerical_columns(schema, _count, _normal):
    assert schema.LABEL_COLUMN not in schema.NUMERICAL_COLUMNS


# ---------------------------------------------------------------------------
# 7. len(CATEGORICAL_COLUMNS) + len(NUMERICAL_COLUMNS) == FEATURE_COUNT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, _normal", SCHEMAS)
def test_column_counts_sum_to_feature_count(schema, _count, _normal):
    total = len(schema.CATEGORICAL_COLUMNS) + len(schema.NUMERICAL_COLUMNS)
    assert total == schema.FEATURE_COUNT


# ---------------------------------------------------------------------------
# 8. BINARY_LABEL_MAP is a dict and the normal label maps to 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("schema, _count, normal_label", SCHEMAS)
def test_binary_label_map_is_dict(schema, _count, normal_label):
    assert isinstance(schema.BINARY_LABEL_MAP, dict)


@pytest.mark.parametrize("schema, _count, normal_label", SCHEMAS)
def test_binary_label_map_normal_label_maps_to_zero(schema, _count, normal_label):
    assert schema.BINARY_LABEL_MAP[normal_label] == 0
