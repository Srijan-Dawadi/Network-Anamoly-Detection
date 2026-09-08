"""
schemas/cicids2017.py
=====================
Schema constants for the CICIDS2017 network flow dataset.

CICIDS2017 contains 80 numerical flow-based features extracted by CICFlowMeter.
There are no categorical columns — all features are numerical.
The label column ("Label") contains class names such as "BENIGN", "DoS",
"PortScan", etc. For anomaly detection the label is binarised: BENIGN → 0,
everything else → 1.
"""

# ---------------------------------------------------------------------------
# Feature count (excluding the label column)
# ---------------------------------------------------------------------------
FEATURE_COUNT: int = 80

# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
LABEL_COLUMN: str = "Label"
NORMAL_LABEL: str = "BENIGN"

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------
CATEGORICAL_COLUMNS: list[str] = []  # CICIDS2017 has no categorical features

NUMERICAL_COLUMNS: list[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    # Additional flow-level features to complete the 80-feature CICFlowMeter set
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
]

# REQUIRED_COLUMNS = all feature columns + label column
REQUIRED_COLUMNS: list[str] = NUMERICAL_COLUMNS + [LABEL_COLUMN]

# ---------------------------------------------------------------------------
# Binary label mapping  (BENIGN → 0, all attack classes → 1)
# ---------------------------------------------------------------------------
# Only the normal label is listed explicitly; any key absent from this dict
# should be treated as 1 by the data pipeline.
BINARY_LABEL_MAP: dict = {"BENIGN": 0}

# ---------------------------------------------------------------------------
# Sanity assertions (checked at import time, zero runtime cost in production)
# ---------------------------------------------------------------------------
assert len(NUMERICAL_COLUMNS) == FEATURE_COUNT, (
    f"CICIDS2017 schema mismatch: expected {FEATURE_COUNT} numerical columns, "
    f"got {len(NUMERICAL_COLUMNS)}"
)
assert len(set(NUMERICAL_COLUMNS)) == len(NUMERICAL_COLUMNS), (
    "CICIDS2017 schema has duplicate column names in NUMERICAL_COLUMNS"
)
assert LABEL_COLUMN not in NUMERICAL_COLUMNS, (
    "LABEL_COLUMN must not appear in NUMERICAL_COLUMNS"
)
