# Network Anomaly Autoencoder

A production-style deep-autoencoder pipeline for unsupervised anomaly detection on
network traffic (NSL-KDD), with strict schema validation, deterministic artifact
tracking, and a property-tested core.

## Quickstart (Windows PowerShell)

From the project root:

```powershell
# 1. Download the real NSL-KDD dataset into data/ (KDDTrain+.csv, ~15 MB)
.\run.ps1 setup

# 2. Train an experiment -> runs/<timestamp>/
.\run.ps1 train

# 3. Classify new records with the most recent run
.\run.ps1 infer                       # uses data\sample_infer.csv
.\run.ps1 infer -Data path\to\data.csv
.\run.ps1 infer -RunDir runs\20260101_120000 -Data path\to\data.csv

# 4. Run the full test suite (188 tests, ~96% coverage)
.\run.ps1 test
```

No environment setup is needed — `run.ps1` sets `PYTHONPATH` for you.

## Manual usage (equivalent commands)

```bash
python scripts/make_data.py                 # download + clean NSL-KDD data
python main.py train --config configs/nsl_kdd_default.yaml
python main.py infer --config configs/nsl_kdd_default.yaml --data data/sample_infer.csv
python main.py infer --config configs/nsl_kdd_default.yaml --data data.csv --run-dir runs/<timestamp>
python -m pytest tests/ --cov=network_anomaly_autoencoder
```

You can also install the package (requires this README, already present):

```bash
pip install -e .[dev]
```

## What a run produces

Training writes a timestamped directory under `runs/` containing:

- `config.yaml` — the exact configuration used
- `dataset_stats.json` — size, normal/attack counts, missing-value summary
- `encoder.pkl` — one-hot encoder for categorical features
- `scaler.pkl` — fitted standard scaler
- `model.json` + `model_weights.keras` — the trained autoencoder
- `history.json` — training loss/accuracy history
- `threshold.json` — the anomaly threshold (95th percentile of normal loss)
- `model_plot.png` — architecture diagram

Inference requires the four inference-critical artifacts
(`model_weights.keras`, `scaler.pkl`, `encoder.pkl`, `threshold.json`); if any is
missing, classification fails fast with a clear error.

## Dataset

`scripts/make_data.py` downloads the real NSL-KDD `KDDTrain+.csv` from a public
mirror. The mirror file is headerless with 43 columns (41 features + label +
difficulty); the script strips the difficulty column and writes a clean,
header-row CSV to `data/KDDTrain+.csv`, plus `data/sample_infer.csv` for quick
demo inference. Run with `--force` to re-download.

## Project layout

```
main.py                                 CLI: train / infer
configs/nsl_kdd_default.yaml            reference experiment configuration
run.ps1                                 PowerShell convenience wrapper
scripts/make_data.py                    NSL-KDD download + cleaning
network_anomaly_autoencoder/
    config.py                           validated config objects
    schemas/nsl_kdd.py                  NSL-KDD schema (41 features / labels)
    data_pipeline.py                    load, validate, split, encode, scale
    model.py                            autoencoder build/train
    inference.py                        artifact-driven thresholding + predict
    experiment_tracker.py               run lifecycle + artifacts + metrics
tests/
    unit/                               targeted unit tests
    property/                           Hypothesis property tests (P1-P18)
    integration/                        end-to-end training + inference (16.1-16.4)
```

## Design requirements

The implementation follows the requirements in
`.kiro/specs/network-anomaly-autoencoder/tasks.md`, which cover data integrity,
schema/split/encode/scaling correctness, model behavior, anomaly-threshold
semantics, evaluation metrics, deterministic experiment tracking, and CLI usage.