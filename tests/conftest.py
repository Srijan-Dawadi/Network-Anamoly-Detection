"""
tests/conftest.py
=================
Shared pytest fixtures used across unit, property, and integration test suites.

All synthetic data generation lives here so individual test modules stay lean
and the dataset construction logic is maintained in one place.
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# NSL-KDD synthetic data helpers
# ---------------------------------------------------------------------------

NSL_KDD_CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NSL_KDD_PROTOCOL_VALUES = ["tcp", "udp", "icmp"]
NSL_KDD_SERVICE_VALUES = [
    "http", "ftp", "smtp", "ssh", "dns", "ftp_data", "other"
]
NSL_KDD_FLAG_VALUES = ["SF", "S0", "REJ", "RSTO", "RSTOS0", "SH", "OTH"]

# 38 numerical feature names for NSL-KDD (excluding the 3 categorical + label)
NSL_KDD_NUMERICAL_COLUMNS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

NSL_KDD_ALL_COLUMNS = (
    NSL_KDD_NUMERICAL_COLUMNS
    + NSL_KDD_CATEGORICAL_COLUMNS
    + ["label"]
)


def _make_nsl_kdd_df(
    n_normal: int = 100,
    n_attack: int = 50,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return a synthetic NSL-KDD-like DataFrame with *n_normal* normal rows
    and *n_attack* attack rows.  All numerical features are drawn from a
    standard normal distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_total = n_normal + n_attack
    data: dict[str, object] = {}

    for col in NSL_KDD_NUMERICAL_COLUMNS:
        data[col] = rng.standard_normal(n_total)

    data["protocol_type"] = rng.choice(NSL_KDD_PROTOCOL_VALUES, size=n_total)
    data["service"] = rng.choice(NSL_KDD_SERVICE_VALUES, size=n_total)
    data["flag"] = rng.choice(NSL_KDD_FLAG_VALUES, size=n_total)
    data["label"] = ["normal"] * n_normal + ["neptune"] * n_attack

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nsl_kdd_df_small() -> pd.DataFrame:
    """Minimal NSL-KDD DataFrame: 100 normal + 50 attack rows."""
    return _make_nsl_kdd_df(n_normal=100, n_attack=50)


@pytest.fixture(scope="session")
def nsl_kdd_df_medium() -> pd.DataFrame:
    """Medium NSL-KDD DataFrame: 300 normal + 100 attack rows."""
    return _make_nsl_kdd_df(n_normal=300, n_attack=100)


@pytest.fixture
def nsl_kdd_csv_path(tmp_path: pytest.TempPathFactory, nsl_kdd_df_small: pd.DataFrame):
    """Write the small synthetic NSL-KDD DataFrame to a temporary CSV and
    return the file path as a string."""
    csv_file = tmp_path / "KDDTrain_synthetic.csv"
    nsl_kdd_df_small.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def nsl_kdd_csv_path_medium(
    tmp_path: pytest.TempPathFactory,
    nsl_kdd_df_medium: pd.DataFrame,
):
    """Write the medium synthetic NSL-KDD DataFrame to a temporary CSV."""
    csv_file = tmp_path / "KDDTrain_medium.csv"
    nsl_kdd_df_medium.to_csv(csv_file, index=False)
    return str(csv_file)
