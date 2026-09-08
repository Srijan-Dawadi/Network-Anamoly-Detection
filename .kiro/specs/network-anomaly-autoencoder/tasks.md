# Implementation Plan: network-anomaly-autoencoder

## Overview

Implement a production-quality deep learning pipeline that detects anomalous network traffic using a deep Autoencoder trained exclusively on normal traffic. The plan follows the modular architecture defined in the design: exceptions → config → schemas → data pipeline → model → threshold → evaluator → experiment tracker → CLI entry point, with a full property-based and unit test suite powered by Hypothesis and pytest.

## Tasks

- [x] 1. Set up project structure, dependencies, and core exception hierarchy
  - Create the `network_anomaly_autoencoder/` package skeleton with `__init__.py`, subdirectory `schemas/`, and `utils/` with `__init__.py` stubs
  - Create `tests/unit/`, `tests/property/`, `tests/integration/` with `__init__.py` and a root `tests/conftest.py`
  - Create `pyproject.toml` (or `requirements.txt`) pinning: `tensorflow>=2.15`, `scikit-learn>=1.4`, `pandas>=2.0`, `numpy>=1.26`, `pyyaml>=6.0`, `matplotlib>=3.8`, `joblib>=1.3`, `hypothesis>=6.100`, `pytest>=8.0`, `pytest-cov>=4.1`
  - Implement `network_anomaly_autoencoder/exceptions.py` with the full custom exception hierarchy: `NetworkAnomalyBaseError`, `DataLoadError`, `SchemaValidationError`, `ConfigurationError`, `InferenceError`, `ArtifactLoadError`, `ArtifactNotFoundError`, `ArtifactWriteError`
  - _Requirements: 1.2, 1.3, 2.4, 3.2, 4.7, 5.4, 9.4_

- [x] 2. Implement configuration dataclasses and YAML loader
  - [x] 2.1 Implement `config.py` with all six dataclasses (`DatasetConfig`, `SplitsConfig`, `ArchitectureConfig`, `TrainingConfig`, `ThresholdConfig`, `OutputConfig`, `ExperimentConfig`) and the `load_config(yaml_path)` function that validates all required top-level keys and value ranges, raising `ConfigurationError` with the offending key on any violation
    - _Requirements: 8.1, 8.5, 3.2, 4.2_

  - [x]* 2.2 Write property test for `load_config` missing-key detection (Property 18)
    - **Property 18: Missing YAML Key Raises ConfigurationError Naming the Key**
    - **Validates: Requirements 8.1, 8.5**

  - [x] 2.3 Write unit tests for `config.py`
    - Test successful round-trip load from a valid YAML
    - Test each of the six top-level keys missing individually
    - Test out-of-range values (e.g. negative learning rate, percentile > 99)
    - _Requirements: 8.1, 8.5_

- [x] 3. Implement dataset schema definitions
  - [x] 3.1 Implement `schemas/nsl_kdd.py` exposing `FEATURE_COUNT`, `LABEL_COLUMN`, `NORMAL_LABEL`, `CATEGORICAL_COLUMNS`, `NUMERICAL_COLUMNS`, `REQUIRED_COLUMNS`, and `BINARY_LABEL_MAP`
    - _Requirements: 1.1_

  - [x] 3.2 Implement `schemas/cicids2017.py` and `schemas/unsw_nb15.py` with matching structure for their respective feature counts (80 and 49)
    - _Requirements: 1.1_

  - [x] 3.3 Write unit tests for all three schema modules asserting correct `FEATURE_COUNT`, non-empty column lists, and that `REQUIRED_COLUMNS` is the union of categorical + numerical + label columns
    - _Requirements: 1.1_

- [x] 4. Implement utility modules
  - [x] 4.1 Implement `utils/logging_utils.py` with a `get_logger(name, log_file=None)` factory that attaches a `StreamHandler` and, when `log_file` is provided, a `FileHandler`; format: `"%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"`
    - _Requirements: 10.4, 10.5_

  - [x] 4.2 Implement `utils/seed_utils.py` with `set_global_seeds(seed: int)` that sets `random.seed`, `numpy.random.seed`, and `tf.random.set_seed` in one call
    - _Requirements: 4.5_

  - [x] 4.3 Write unit tests for `logging_utils` asserting timestamp, level, and module fields appear in formatted output, and that a `FileHandler` is attached when `log_file` is supplied
    - _Requirements: 10.4, 10.5_

- [x] 5. Implement `Data_Pipeline` — loading, validation, and deduplication
  - [x] 5.1 Implement `Data_Pipeline.__init__`, `load_and_validate`, `_deduplicate`, and `_report_counts` in `data_pipeline.py`
    - `load_and_validate` must raise `DataLoadError` (with file path) for missing/unreadable files and for zero-record files; raise `SchemaValidationError` listing all missing columns if schema columns are absent
    - `_deduplicate` removes duplicate rows (all feature + label columns identical) and logs count at INFO
    - `_report_counts` logs total, normal, and attack record counts
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x]* 5.2 Write property test for schema column-count validation (Property 1)
    - **Property 1: Schema Column-Count Validation**
    - **Validates: Requirements 1.1, 1.3**

  - [x]* 5.3 Write property test for missing-file `DataLoadError` (Property 2)
    - **Property 2: Missing File Raises DataLoadError with Path**
    - **Validates: Requirements 1.2**

  - [x]* 5.4 Write property test for deduplication idempotence (Property 3)
    - **Property 3: Deduplication Idempotence**
    - **Validates: Requirements 1.5**

  - [x]* 5.5 Write property test for total = normal + attack count invariant (Property 4)
    - **Property 4: Count Invariant — Total Equals Normal Plus Attack**
    - **Validates: Requirements 1.4**

  - [x] 5.6 Write unit tests for `load_and_validate`
    - Test missing file path raises `DataLoadError` with path in message
    - Test zero-record file raises `DataLoadError` indicating empty file
    - Test file with missing columns raises `SchemaValidationError` listing them
    - Test successful load on minimal synthetic NSL-KDD CSV
    - _Requirements: 1.2, 1.3, 1.6_

- [x] 6. Implement `Data_Pipeline` — preprocessing, splitting, and artefact I/O
  - [x] 6.1 Implement `fit_transform`, `_validate_split_ratios`, `save_artefacts`, `load_artefacts`, and `transform` in `data_pipeline.py`
    - `fit_transform` one-hot encodes categoricals with `sklearn.preprocessing.OneHotEncoder` (handle_unknown='ignore'), applies `StandardScaler` to numericals, splits by ratio with fixed seed producing (X_train_normal, X_val_normal, X_test_mixed, y_train, y_val, y_test)
    - `_validate_split_ratios` raises `ConfigurationError` naming invalid ratio when any ratio ≤ 0 or ≥ 1 or sum ≠ 1.0
    - `save_artefacts` / `load_artefacts` persist and restore OHE + scaler as `.pkl` via `joblib`
    - `transform` raises `InferenceError` if artefacts not loaded; unseen categorical values silently encoded as all-zero vector
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x]* 6.2 Write property test for OHE and feature dimensionality consistency (Property 5)
    - **Property 5: OHE and Feature Dimensionality Consistency**
    - **Validates: Requirements 2.1, 2.5**

  - [x]* 6.3 Write property test for StandardScaler normalisation invariant (Property 6)
    - **Property 6: StandardScaler Normalisation Invariant**
    - **Validates: Requirements 2.2**

  - [x]* 6.4 Write property test for unknown categorical value encoded as all-zero vector (Property 7)
    - **Property 7: Unknown Categorical Value Encoded as All-Zero Vector**
    - **Validates: Requirements 2.3**

  - [x]* 6.5 Write property test for split-ratio validity and size proportionality (Property 8)
    - **Property 8: Split-Ratio Validity and Size Proportionality**
    - **Validates: Requirements 2.4**

  - [x] 6.6 Write unit tests for preprocessing
    - Test `ConfigurationError` on invalid ratios (zero, negative, sum ≠ 1.0)
    - Test `DataLoadError` when fewer than 2 normal records after split
    - Test training/val/test splits contain only normal / only normal / mixed samples as required
    - Test `InferenceError` raised by `transform` when artefacts not loaded
    - _Requirements: 2.4, 2.6_

- [x] 7. Checkpoint — verify data pipeline tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement `Autoencoder_Model` — architecture construction
  - [x] 8.1 Implement `Autoencoder_Model.__init__`, `build`, `_build_lr_schedule`, and `_set_seeds` in `model.py`
    - `build(input_dim)` constructs the symmetric encoder-decoder graph: `Input → [Dense → BatchNorm → (Dropout)?]×n → Bottleneck → [Dense → BatchNorm → (Dropout)?]×n → Dense(input_dim, linear)`; compiles with Adam + MSE loss
    - Dropout layers are added if and only if `dropout_rate > 0`
    - Raises `ConfigurationError` naming the invalid parameter for any out-of-range arch config value
    - `_build_lr_schedule` returns the appropriate Keras LR schedule for `"step_decay"`, `"cosine_annealing"`, or `None` for `"none"`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x]* 8.2 Write property test for symmetric architecture with BatchNorm and Dropout (Property 9)
    - **Property 9: Autoencoder Symmetric Architecture with BatchNorm and Dropout**
    - **Validates: Requirements 3.1, 3.3, 3.4**

  - [x]* 8.3 Write property test for `ConfigurationError` on out-of-range parameters (Property 10)
    - **Property 10: ConfigurationError for Out-of-Range Parameters**
    - **Validates: Requirements 3.2, 4.2**

  - [x] 8.4 Write unit tests for `build`
    - Test `ConfigurationError` for each invalid arch param (bottleneck ≥ min layer, layer < 1, dropout ≥ 1.0)
    - Test no `Dropout` layers when `dropout_rate == 0`
    - Test MSE is the compiled loss function
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [x] 9. Implement `Autoencoder_Model` — training and reconstruction
  - [x] 9.1 Implement `fit`, `reconstruct`, `predict`, `save_weights`, and `load_weights` in `model.py`
    - `fit` trains on normal-traffic `X_train`, monitors `X_val`, applies `EarlyStopping(patience=patience, restore_best_weights=True)`, saves best weights, logs epoch loss at DEBUG; raises `DataLoadError` on empty `X_train`; raises `ArtifactWriteError` on unwritable path
    - `reconstruct` returns `(X_hat, per_sample_mse)` with matching shape; raises `InferenceError` if model not built
    - `predict` returns `(binary_preds, mse_scores)` with threshold comparison `mse >= threshold`; raises `InferenceError` on feature dimension mismatch or empty batch
    - _Requirements: 3.6, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 6.1, 6.2, 6.3, 6.5, 6.6_

  - [x]* 9.2 Write property test for `reconstruct(X)` output shape invariant (Property 11)
    - **Property 11: reconstruct(X) Output Shape Invariant**
    - **Validates: Requirements 3.6**

  - [x]* 9.3 Write property test for `predict(X)` binary threshold classification correctness (Property 12)
    - **Property 12: predict(X) Binary Threshold Classification Correctness**
    - **Validates: Requirements 6.1, 6.2, 6.5**

  - [x] 9.4 Write unit tests for training and inference
    - Test `DataLoadError` raised when `X_train` is empty
    - Test `ArtifactWriteError` raised for unwritable save path
    - Test early stopping halts before `max_epochs` on overfit synthetic data
    - Test `InferenceError` raised by `reconstruct` before `build` is called
    - Test `InferenceError` raised by `predict` on feature dimension mismatch
    - Test `InferenceError` raised by `predict` on empty batch
    - _Requirements: 3.6, 4.3, 4.6, 4.7, 6.3, 6.5, 6.6_

- [x] 10. Implement `Threshold_Estimator`
  - [x] 10.1 Implement `Threshold_Estimator` in `threshold.py` including `fit`, `save`, `load`, and the `threshold` property
    - `fit` computes `np.percentile(mse_scores, percentile)`, logs threshold, percentile, mean, std at INFO, and stores all four statistics
    - `save` writes `{ threshold, percentile, mse_mean, mse_std }` to `<run_dir>/threshold.json`
    - `load` reads the JSON, validates all four fields are present and parseable, returns the threshold value; raises `ArtifactLoadError` naming missing/unparseable fields
    - `threshold` property raises `InferenceError` if not yet computed or loaded
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x]* 10.2 Write property test for threshold round-trip (Property 13)
    - **Property 13: Threshold Round-Trip — Compute, Save, Load**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

  - [x] 10.3 Write unit tests for `Threshold_Estimator`
    - Test `ArtifactLoadError` when JSON is missing any required field
    - Test `ArtifactLoadError` when file is not valid JSON
    - Test `InferenceError` from `threshold` property before `fit` or `load`
    - Test percentile boundary values (p=1, p=99)
    - _Requirements: 5.4, 5.5_

- [x] 11. Implement `Evaluator`
  - [x] 11.1 Implement `Evaluator` in `evaluator.py` including `evaluate`, `_compute_classification_metrics`, `_compute_auc_roc`, `_plot_confusion_matrix`, `_plot_roc_curve`, `_plot_re_histogram`, and `_write_metrics_json`
    - `evaluate` raises `ValueError` identifying mismatched lengths before computing any metric
    - `_compute_classification_metrics` returns precision/recall/F1/accuracy for both classes
    - `_compute_auc_roc` uses `mse_scores` as ranking signal; emits WARNING when AUC-ROC < 0.80
    - All three plots saved as PNG artefacts; metrics JSON written with all eight required keys
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [x]* 11.2 Write property test for evaluator metrics validity and JSON completeness (Property 14)
    - **Property 14: Evaluator Metrics Validity and JSON Completeness**
    - **Validates: Requirements 7.1, 7.2, 7.6**

  - [x] 11.3 Write unit tests for `Evaluator`
    - Test `ValueError` on length-mismatched inputs
    - Test all three PNG files are created in the run directory
    - Test `metrics.json` contains all eight required keys with values in [0.0, 1.0]
    - Test WARNING log emitted when AUC-ROC < 0.80 (mock the AUC return)
    - _Requirements: 7.1, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 12. Checkpoint — verify model, threshold, and evaluator tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement `Experiment_Tracker`
  - [x] 13.1 Implement `Experiment_Tracker.__init__`, `setup`, `_create_run_dir`, `_log_git_hash`, and `_record_timing` in `experiment_tracker.py`
    - `setup` calls `load_config`, creates `<artefact_dir>/YYYYMMDD_HHMMSS/` subdirectory, sets up console + file logger (using `logging_utils`), copies YAML byte-for-byte, logs git hash (or `"git-unavailable"`)
    - `_create_run_dir` produces the correct timestamp format
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 10.1, 10.4, 10.5_

  - [x]* 13.2 Write property test for config file byte-for-byte copy (Property 17)
    - **Property 17: Config File Byte-for-Byte Copy**
    - **Validates: Requirements 8.2**

  - [x] 13.3 Write unit tests for `setup`
    - Test `ConfigurationError` raised for missing YAML keys before data loading
    - Test run directory created with correct `YYYYMMDD_HHMMSS` pattern
    - Test YAML copied byte-for-byte into the run directory
    - Test `"git-unavailable"` logged when not inside a git repo
    - Test log file created inside run directory
    - _Requirements: 8.1, 8.2, 8.3, 8.5, 9.1, 10.5_

- [x] 14. Implement `Experiment_Tracker` — training and inference orchestration
  - [x] 14.1 Implement `run_training` in `experiment_tracker.py`
    - Orchestrates: `setup` → `Data_Pipeline.load_and_validate` → `fit_transform` → `Autoencoder_Model.build` → `fit` → `Threshold_Estimator.fit` → `Evaluator.evaluate` → `save_artefacts` + `save_weights` + `Threshold_Estimator.save`
    - Logs INFO at start and end of each major phase; catches all exceptions at phase boundaries, logs type + message + traceback at ERROR, then re-raises
    - Records wall-clock training time and writes to `metrics.json`
    - _Requirements: 9.2, 10.1, 10.2, 8.4_

  - [x] 14.2 Implement `run_inference` in `experiment_tracker.py`
    - Pre-checks existence of all four critical artefact files before loading any; raises `ArtifactNotFoundError` listing all missing files if any are absent
    - Orchestrates: `Data_Pipeline.load_artefacts` → `transform` → `Autoencoder_Model.load_weights` → `Threshold_Estimator.load` → `predict`
    - Logs INFO at start and end of inference phase
    - _Requirements: 9.3, 9.4, 10.1_

  - [x]* 14.3 Write property test for `ArtifactNotFoundError` listing all missing files (Property 15)
    - **Property 15: ArtifactNotFoundError Lists All Missing Files**
    - **Validates: Requirements 9.4**

  - [x]* 14.4 Write property test for artefact completeness after training (Property 16)
    - **Property 16: Artefact Completeness After Training**
    - **Validates: Requirements 9.2**

  - [x] 14.5 Write unit tests for `run_training` and `run_inference`
    - Test all nine artefact files present after `run_training` on minimal synthetic data
    - Test ERROR-level log with traceback emitted when a component raises
    - Test `ArtifactNotFoundError` raised when any single critical artefact is missing before inference
    - Test wall-clock time written to `metrics.json`
    - _Requirements: 8.4, 9.2, 9.3, 9.4, 10.2_

- [x] 15. Create reference YAML configuration and `main.py` CLI entry point
  - [x] 15.1 Create `configs/nsl_kdd_default.yaml` with the reference values from the design (`encoder_layers: [128,64,32]`, `bottleneck_dim: 16`, `dropout_rate: 0.2`, `percentile: 95`, etc.)
    - _Requirements: 8.1_

  - [x] 15.2 Implement `main.py` with a CLI (`argparse`) exposing two sub-commands: `train --config <yaml>` and `infer --config <yaml> --data <csv>`
    - `train` instantiates `Experiment_Tracker` and calls `run_training`
    - `infer` instantiates `Experiment_Tracker` and calls `run_inference`
    - Catches `NetworkAnomalyBaseError` at top level, prints the error to stderr, and exits with code 1
    - _Requirements: 8.1_

  - [x] 15.3 Write unit tests for `main.py`
    - Test exit code 1 and stderr output on `NetworkAnomalyBaseError`
    - Test `train` sub-command routes to `run_training`
    - Test `infer` sub-command routes to `run_inference`
    - _Requirements: 8.1_

- [x] 16. Write integration tests
  - [x]* 16.1 Write end-to-end training integration test
    - Generate a synthetic NSL-KDD-like CSV (~300 normal rows, 100 attack rows) using `numpy.random`
    - Run `run_training` against this data; assert all nine artefact files exist in the run directory; assert `metrics.json` contains all eight required metric keys; assert all metric values are in [0.0, 1.0]
    - _Requirements: 9.2, 7.6_

  - [x]* 16.2 Write inference-from-artefact integration test
    - Load artefacts from the training run in 16.1; call `run_inference` on a synthetic batch; assert predictions shape equals input row count; assert no exceptions raised
    - _Requirements: 9.3_

  - [x]* 16.3 Write CPU inference timing integration test
    - Generate 10,000 synthetic feature rows; run `predict` on CPU (no GPU); assert wall-clock time < 5 seconds
    - _Requirements: 6.4_

  - [x]* 16.4 Write early-stopping integration test
    - Configure `max_epochs: 200`, `patience: 3` on intentionally overfit data; assert training stops before epoch 200
    - _Requirements: 4.3_

- [x] 17. Final checkpoint — ensure all tests pass
  - Run `pytest tests/ --cov=network_anomaly_autoencoder --cov-report=term-missing` and confirm ≥ 90% line coverage on all non-UI code; ensure all property, unit, and integration tests pass; ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; all 18 correctness properties are covered by the `*`-marked property test sub-tasks
- Each task references specific requirements for traceability
- Checkpoints (tasks 7, 12, 17) validate incremental progress before moving to the next phase
- The design uses Python with Hypothesis for property-based testing and pytest as the test runner
- Synthetic data fixtures in `tests/conftest.py` are reused across unit, property, and integration tests to avoid external dataset downloads
- The `runs/` directory should be added to `.gitignore` to avoid committing large artefacts

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1", "3.2", "4.1", "4.2"] },
    { "id": 1, "tasks": ["2.2", "2.3", "3.3", "4.3", "5.1", "6.1"] },
    { "id": 2, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "6.2", "6.3", "6.4", "6.5", "6.6"] },
    { "id": 3, "tasks": ["8.1", "10.1", "11.1"] },
    { "id": 4, "tasks": ["8.2", "8.3", "8.4", "10.2", "10.3", "11.2", "11.3", "9.1"] },
    { "id": 5, "tasks": ["9.2", "9.3", "9.4", "13.1"] },
    { "id": 6, "tasks": ["13.2", "13.3", "14.1", "14.2"] },
    { "id": 7, "tasks": ["14.3", "14.4", "14.5", "15.1", "15.2"] },
    { "id": 8, "tasks": ["15.3", "16.1", "16.2", "16.3", "16.4"] }
  ]
}
```
