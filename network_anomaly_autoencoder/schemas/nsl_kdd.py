"""
schemas/nsl_kdd.py
==================
Schema constants for the NSL-KDD network intrusion detection dataset.

NSL-KDD contains 41 features: 3 categorical + 38 numerical, plus a label column.
The label column ("label") contains class names such as "normal", "neptune",
"smurf", etc. For anomaly detection the label is binarised: normal → 0,
all attack sub-categories → 1.
"""

# ---------------------------------------------------------------------------
# Feature count (excluding the label column)
# ---------------------------------------------------------------------------
FEATURE_COUNT: int = 41

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
LABEL_COLUMN: str = "label"
NORMAL_LABEL: str = "normal"

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS: list[str] = [
    "protocol_type",
    "service",
    "flag",
]

NUMERICAL_COLUMNS: list[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# REQUIRED_COLUMNS = categorical + numerical + label
REQUIRED_COLUMNS: list[str] = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [LABEL_COLUMN]

# ---------------------------------------------------------------------------
# Binary label mapping  (normal → 0, all known attack sub-categories → 1)
# ---------------------------------------------------------------------------
BINARY_LABEL_MAP: dict = {
    "normal": 0,
    # DoS attacks
    "neptune": 1,
    "smurf": 1,
    "back": 1,
    "teardrop": 1,
    "pod": 1,
    "land": 1,
    # Probe attacks
    "ipsweep": 1,
    "portsweep": 1,
    "satan": 1,
    "mscan": 1,
    "saint": 1,
    # R2L attacks
    "warezclient": 1,
    "guess_passwd": 1,
    "warezmaster": 1,
    "imap": 1,
    "ftp_write": 1,
    "multihop": 1,
    "phf": 1,
    "spy": 1,
    "sendmail": 1,
    "named": 1,
    "snmpgetattack": 1,
    "snmpguess": 1,
    "xlock": 1,
    "xsnoop": 1,
    "worm": 1,
    # U2R attacks
    "rootkit": 1,
    "buffer_overflow": 1,
    "loadmodule": 1,
    "perl": 1,
    "xterm": 1,
    "ps": 1,
    "sqlattack": 1,
    "httptunnel": 1,
    "processtable": 1,
    "udpstorm": 1,
    "apache2": 1,
    # Generic category labels (sometimes present in KDDTest+)
    "dos": 1,
    "probe": 1,
    "r2l": 1,
    "u2r": 1,
}

# ---------------------------------------------------------------------------
# Sanity assertions (checked at import time, zero runtime cost in production)
# ---------------------------------------------------------------------------
assert len(CATEGORICAL_COLUMNS) + len(NUMERICAL_COLUMNS) == FEATURE_COUNT, (
    f"NSL-KDD schema mismatch: expected {FEATURE_COUNT} feature columns, "
    f"got {len(CATEGORICAL_COLUMNS) + len(NUMERICAL_COLUMNS)}"
)
assert len(NUMERICAL_COLUMNS) == 38, (
    f"NSL-KDD schema mismatch: expected 38 numerical columns, "
    f"got {len(NUMERICAL_COLUMNS)}"
)
assert len(CATEGORICAL_COLUMNS) == 3, (
    f"NSL-KDD schema mismatch: expected 3 categorical columns, "
    f"got {len(CATEGORICAL_COLUMNS)}"
)
assert len(set(NUMERICAL_COLUMNS)) == len(NUMERICAL_COLUMNS), (
    "NSL-KDD schema has duplicate column names in NUMERICAL_COLUMNS"
)
assert len(set(CATEGORICAL_COLUMNS)) == len(CATEGORICAL_COLUMNS), (
    "NSL-KDD schema has duplicate column names in CATEGORICAL_COLUMNS"
)
assert LABEL_COLUMN not in NUMERICAL_COLUMNS, (
    "LABEL_COLUMN must not appear in NUMERICAL_COLUMNS"
)
assert LABEL_COLUMN not in CATEGORICAL_COLUMNS, (
    "LABEL_COLUMN must not appear in CATEGORICAL_COLUMNS"
)
