"""
schemas/unsw_nb15.py
====================
Schema constants for the UNSW-NB15 network intrusion detection dataset.

UNSW-NB15 contains 49 features: 3 categorical + 45 numerical, plus a label column.
The label column ("label") is an integer: 0 = normal, 1 = attack.
For anomaly detection the label is used directly as the binary target.
"""

# ---------------------------------------------------------------------------
# Feature count (excluding the label column)
# ---------------------------------------------------------------------------
FEATURE_COUNT: int = 49

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
LABEL_COLUMN: str = "label"
NORMAL_LABEL: int = 0  # 0 = normal, 1 = attack

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS: list[str] = [
    "proto",
    "service",
    "state",
]

NUMERICAL_COLUMNS: list[str] = [
    # Port / timing
    "sport",          # 1
    "dport",          # 2
    "dur",            # 3
    "Stime",          # 4
    "Ltime",          # 5
    # Packet / byte counts
    "spkts",          # 6
    "dpkts",          # 7
    "sbytes",         # 8
    "dbytes",         # 9
    # TTL
    "sttl",           # 10
    "dttl",           # 11
    # Loss
    "sloss",          # 12
    "dloss",          # 13
    # Load
    "Sload",          # 14
    "Dload",          # 15
    # Window sizes
    "swin",           # 16
    "dwin",           # 17
    # TCP base sequence numbers
    "stcpb",          # 18
    "dtcpb",          # 19
    # Mean segment sizes
    "smeansz",        # 20
    "dmeansz",        # 21
    # Application-layer depth / body length
    "trans_depth",    # 22
    "res_bdy_len",    # 23
    # Jitter
    "Sjit",           # 24
    "Djit",           # 25
    # Inter-packet timing
    "Sintpkt",        # 26
    "Dintpkt",        # 27
    # TCP round-trip / handshake
    "tcprtt",         # 28
    "synack",         # 29
    "ackdat",         # 30
    # Binary flags
    "is_sm_ips_ports",    # 31
    # Connection-state TTL aggregation
    "ct_state_ttl",       # 32
    # HTTP methods count
    "ct_flw_http_mthd",   # 33
    # FTP login / command features
    "is_ftp_login",       # 34
    "ct_ftp_cmd",         # 35
    # Connection-table aggregates
    "ct_srv_src",         # 36
    "ct_srv_dst",         # 37
    "ct_dst_ltm",         # 38
    "ct_src_ltm",         # 39
    "ct_src_dport_ltm",   # 40
    "ct_dst_sport_ltm",   # 41
    "ct_dst_src_ltm",     # 42
    # Additional aggregates to reach 46 (so 3 cat + 46 num = FEATURE_COUNT 49)
    "ct_src_ltm2",        # 43
    "ct_dst_src_ltm2",    # 44
    "ct_srv_dst2",        # 45
    "ct_dst_ltm2",        # 46
]

# REQUIRED_COLUMNS = categorical + numerical + label
REQUIRED_COLUMNS: list[str] = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [LABEL_COLUMN]

# ---------------------------------------------------------------------------
# Binary label mapping  (0 = normal, 1 = attack — already binary in UNSW-NB15)
# ---------------------------------------------------------------------------
BINARY_LABEL_MAP: dict = {0: 0, 1: 1}

# ---------------------------------------------------------------------------
# Sanity assertions (checked at import time, zero runtime cost in production)
# ---------------------------------------------------------------------------
assert len(CATEGORICAL_COLUMNS) + len(NUMERICAL_COLUMNS) == FEATURE_COUNT, (
    f"UNSW-NB15 schema mismatch: expected {FEATURE_COUNT} feature columns, "
    f"got {len(CATEGORICAL_COLUMNS) + len(NUMERICAL_COLUMNS)}"
)
assert len(NUMERICAL_COLUMNS) == 46, (
    f"UNSW-NB15 schema mismatch: expected 46 numerical columns, "
    f"got {len(NUMERICAL_COLUMNS)}"
)
assert len(CATEGORICAL_COLUMNS) == 3, (
    f"UNSW-NB15 schema mismatch: expected 3 categorical columns, "
    f"got {len(CATEGORICAL_COLUMNS)}"
)
assert len(set(NUMERICAL_COLUMNS)) == len(NUMERICAL_COLUMNS), (
    "UNSW-NB15 schema has duplicate column names in NUMERICAL_COLUMNS"
)
assert len(set(CATEGORICAL_COLUMNS)) == len(CATEGORICAL_COLUMNS), (
    "UNSW-NB15 schema has duplicate column names in CATEGORICAL_COLUMNS"
)
assert LABEL_COLUMN not in NUMERICAL_COLUMNS, (
    "LABEL_COLUMN must not appear in NUMERICAL_COLUMNS"
)
assert LABEL_COLUMN not in CATEGORICAL_COLUMNS, (
    "LABEL_COLUMN must not appear in CATEGORICAL_COLUMNS"
)
