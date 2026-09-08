"""
scripts/make_data.py
====================
Download the real NSL-KDD training dataset and prepare it for the pipeline.

The KDDTrain+ distribution is a headerless CSV with 43 columns:
41 features, the ``label`` column, and a trailing ``difficulty`` column
present in most mirrors.  This script:

  * downloads ``KDDTrain+.csv`` from a public mirror (unless already present),
  * drops the ``difficulty`` column,
  * writes a clean, header-row CSV to ``data/KDDTrain+.csv``,
  * draws a subset for ``data/sample_infer.csv`` (used by ``run.ps1 infer``).

Usage
-----
    python scripts/make_data.py                 # download if needed, prepare
    python scripts/make_data.py --force         # re-download from scratch
    python scripts/make_data.py --infer-rows 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ULR_DEFAULT = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Canonical 43-column header of the headerless KDDTrain+ distribution.
NSL_KDD_43_COLUMNS = [
    # --- 41 features -------------------------------------------------
    "duration",
    "protocol_type",
    "service",
    "flag",
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
    # --- label + difficulty (43rd dropped) ---------------------------
    "label",
    "difficulty",
]

FEATURE_HEADER = NSL_KDD_43_COLUMNS[:42]
CLEAN_HEADER = NSL_KDD_43_COLUMNS[:-1]


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, streaming to avoid holding it in memory."""
    import urllib.request

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Downloaded {dest.name} ({size_mb:.1f} MB).")


def _prepare(raw: Path) -> pd.DataFrame:
    """Return a clean (41 features + label) DataFrame from the raw file."""
    df = pd.read_csv(raw, header=None, names=NSL_KDD_43_COLUMNS)
    if len(df.columns) > 43:
        raise SystemExit(
            f"Unexpected column count {len(df.columns)} in {raw.name} "
            f"(expected 43). Mirror may have changed."
        )
    df = df[CLEAN_HEADER].copy()

    # Sanity checks on the prepared data.
    if len(df) == 0:
        raise SystemExit("Prepared dataset is empty.")
    if df["label"].isna().any():
        raise SystemExit("Label column contains missing values.")
    labels = set(df["label"].unique())
    if "normal" not in labels or not labels - {"normal"}:
        raise SystemExit(
            f"Expected both 'normal' and attack labels, got: {sorted(labels)[:10]}"
        )

    # NSL-KDD uses '0' for absent interactive service markers; no NaN expected.
    if df.isna().any().any():
        raise SystemExit("Prepared dataset contains missing values.")
    return df


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--url", default=ULR_DEFAULT, help="Mirror URL for KDDTrain+.csv.")
    parser.add_argument("--infer-rows", type=int, default=500, help="Rows for sample_infer.csv.")
    args = parser.parse_args(argv)

    if args.infer_rows < 1:
        parser.error("--infer-rows must be >= 1")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = DATA_DIR / "KDDTrain+_raw.csv"
    out_file = DATA_DIR / "KDDTrain+.csv"
    infer_file = DATA_DIR / "sample_infer.csv"

    # ---- Download (if needed) ---------------------------------------
    if not raw_file.exists() or args.force:
        _download(args.url, raw_file)
    else:
        print(f"Using existing {raw_file.name} (use --force to re-download).")

    # ---- Clean + validate -------------------------------------------
    df = _prepare(raw_file)
    df.to_csv(out_file, index=False)
    print(f"Wrote {out_file.name}: {len(df):,} rows x {df.shape[1]} cols.")

    # ---- Inference sample -------------------------------------------
    n_infer = min(args.infer_rows, len(df))
    infer = df.sample(n=n_infer, random_state=42).reset_index(drop=True)
    infer.to_csv(infer_file, index=False)
    print(f"Wrote {infer_file.name}: {len(infer):,} rows.")

    normal = int((df["label"] == "normal").sum())
    print(f"Summary: {len(df):,} rows total, {normal:,} normal, {len(df) - normal:,} attacks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())