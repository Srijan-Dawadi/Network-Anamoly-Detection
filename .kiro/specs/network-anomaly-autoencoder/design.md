# Design Document — network-anomaly-autoencoder

## Overview

The network-anomaly-autoencoder system is a production-quality deep learning pipeline that detects anomalous network traffic using a deep Autoencoder neural network. The Autoencoder is trained exclusively on **normal** network flow records and learns a compressed latent representation of benign behaviour. At inference time, records that cannot be faithfully reconstructed — those whose reconstruction error (MSE) exceeds a learned threshold — are flagged as anomalies.

### Primary Dataset
The system targets **NSL-KDD** (41 features, cleaned KDD Cup 99 derivative) as its primary dataset. Adapter support for **CICIDS2017** (80 features) and **UNSW-NB15** (49 features) is built in through dataset schema configuration.

### Design Goals
- **Modularity**: each responsibility (loading, preprocessing, model, threshold, evaluation, experiment management) lives in its own Python module with a clean public interface.
- **Reproducibility**: YAML-driven configuration, fixed random seeds, artefact versioning, and Git hash logging.
- **Observability**: structured, levelled logging to both console and file at every major processing step.
- **Correctness**: custom exception hierarchy makes failure modes explicit; property-based testing validates universally quantified behavioural invariants.

---

## Architecture

### High-Level Component Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        main.py / CLI Entry Point                     │
│  (Orchestrates training run or inference run from YAML config)       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │            Experiment_Tracker               │
          │  - reads YAML config                        │
          │  - creates timestamped artefact directory   │
          │  - logs git hash, wall-clock timing         │
          │  - wires components together                │
          └──┬──────────┬──────────┬───────────┬───────┘
             │          │          │           │
     ┌───────▼──┐ ┌─────▼────┐ ┌──▼──────┐ ┌─▼──────────┐
     │  Data_   │ │Autoencoder│ │Threshold│ │ Evaluator  │
     │ Pipeline │ │  _Model   │ │Estimator│ │            │
     └──┬───────┘ └──┬───────┘ └──┬──────┘ └─┬──────────┘
        │            │            │           │
        │            │            │           │
     ┌──▼────────────▼────────────▼───────────▼──────────┐
     │              Artefact Directory                    │
     │  runs/YYYYMMDD_HHMMSS/                             │
     │    model_weights.keras   encoder.pkl               │
     │    scaler.pkl            threshold.json            │
     │    confusion_matrix.png  roc_curve.png             │
     │    re_histogram.png      metrics.json              │
     │    config.yaml           experiment.log            │
     └────────────────────────────────────────────────────┘
```

### Data Flow

```
              TRAINING FLOW
              ─────────────
Raw CSV  ──► Data_Pipeline ──► (X_train_normal, X_val_normal, X_test)
                │
                ├── fitted OHE ──────────────────────────► artefact dir
                └── fitted Scaler ────────────────────────► artefact dir
                          │
                     X_train_normal ──► Autoencoder_Model.fit()
                                              │
                                              ├── best weights ─────────► artefact dir
                                              └── training MSE vector
                                                        │
                                              Threshold_Estimator.fit()
                                                        │
                                                        └── threshold.json ─► artefact dir
                                                                  │
                                                       X_test ──► Autoencoder_Model.predict()
                                                                  │
                                                          (preds, mse_scores)
                                                                  │
                                                       Evaluator.evaluate()
                                                                  │
                                                    metrics.json + PNG artefacts ─► artefact dir

              INFERENCE FLOW
              ──────────────
Raw CSV ──► Data_Pipeline (inference mode, loads fitted OHE + Scaler from artefact dir)
                    │
               X_new ──► Autoencoder_Model.predict()  (loads weights + threshold from artefact dir)
                              │
                     (predictions, mse_scores) ──► caller / downstream system
```

### Module Structure

```
network_anomaly_autoencoder/
├── __init__.py
├── exceptions.py          # Custom exception hierarchy
├── config.py              # Config dataclasses and YAML loader
├── data_pipeline.py       # Data_Pipeline class
├── model.py               # Autoencoder_Model class
├── threshold.py           # Threshold_Estimator class
├── evaluator.py           # Evaluator class
├── experiment_tracker.py  # Experiment_Tracker class
├── schemas/
│   ├── nsl_kdd.py         # NSL-KDD column list and schema constants
│   ├── cicids2017.py      # CICIDS2017 schema
│   └── unsw_nb15.py       # UNSW-NB15 schema
└── utils/
    ├── logging_utils.py   # Logger factory
    └── seed_utils.py      # Global seed setter

tests/
├── unit/
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   ├── test_threshold.py
│   ├── test_evaluator.py
│   └── test_experiment_tracker.py
├── property/
│   ├── test_pipeline_properties.py
│   ├── test_model_properties.py
│   ├── test_threshold_properties.py
│   └── test_evaluator_properties.py
└── integration/
    └── test_end_to_end.py

configs/
└── nsl_kdd_default.yaml   # Reference YAML configuration

runs/                      # Timestamped artefact directories (gitignored)
```

---

## Components and Interfaces

### 1. `exceptions.py` — Exception Hierarchy

```python
class NetworkAnomalyBaseError(Exception):
    """Base exception for all project errors."""

class DataLoadError(NetworkAnomalyBaseError):
    """Raised when a dataset file cannot be loaded or is structurally empty."""

class SchemaValidationError(NetworkAnomalyBaseError):
    """Raised when required columns are absent from the loaded dataset."""

class ConfigurationError(NetworkAnomalyBaseError):
    """Raised when a configuration parameter is invalid or a required key is missing."""

class InferenceError(NetworkAnomalyBaseError):
    """Raised when inference is attempted in an invalid state."""

class ArtifactLoadError(NetworkAnomalyBaseError):
    """Raised when a saved artefact cannot be loaded or is corrupt."""

class ArtifactNotFoundError(NetworkAnomalyBaseError):
    """Raised when expected artefact files are absent from an artefact directory."""

class ArtifactWriteError(NetworkAnomalyBaseError):
    """Raised when a file path for saving an artefact is not writable."""
```

---

### 2. `config.py` — Configuration Dataclasses

```python
@dataclass
class DatasetConfig:
    path: str                 # Absolute/relative path to CSV file
    schema: str               # One of: "nsl_kdd", "cicids2017", "unsw_nb15"

@dataclass
class SplitsConfig:
    train_ratio: float        # (0.0, 1.0) exclusive; normal traffic only
    val_ratio: float          # (0.0, 1.0) exclusive; normal traffic only
    test_ratio: float         # (0.0, 1.0) exclusive; mixed normal + attack
    random_seed: int

@dataclass
class ArchitectureConfig:
    encoder_layers: list[int] # e.g. [128, 64, 32]; each value ≥ 1
    bottleneck_dim: int        # ≥ 1; < min(encoder_layers)
    activation: str            # e.g. "relu", "tanh"
    dropout_rate: float        # [0.0, 1.0)

@dataclass
class TrainingConfig:
    learning_rate: float      # > 0.0
    batch_size: int            # ≥ 1
    max_epochs: int            # ≥ 1
    patience: int              # ≥ 1 (early-stopping patience)
    lr_schedule: str           # "none" | "step_decay" | "cosine_annealing"
    random_seed: int

@dataclass
class ThresholdConfig:
    percentile: int            # [1, 99] inclusive; default 95

@dataclass
class OutputConfig:
    artefact_dir: str          # Base directory; timestamped subdir created at runtime

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    splits: SplitsConfig
    architecture: ArchitectureConfig
    training: TrainingConfig
    threshold: ThresholdConfig
    output: OutputConfig

def load_config(yaml_path: str) -> ExperimentConfig:
    """
    Load and validate YAML configuration file.
    Raises ConfigurationError for missing keys or invalid values.
    """
```

---

### 3. `schemas/` — Dataset Schema Definitions

Each schema module exposes:

```python
# Example: schemas/nsl_kdd.py
FEATURE_COUNT: int = 41
LABEL_COLUMN: str = "label"
NORMAL_LABEL: str = "normal"
CATEGORICAL_COLUMNS: list[str] = ["protocol_type", "service", "flag"]
NUMERICAL_COLUMNS: list[str]   # All 38 non-categorical feature columns
REQUIRED_COLUMNS: list[str]    # CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [LABEL_COLUMN]
```

---

### 4. `data_pipeline.py` — Data_Pipeline

```python
class Data_Pipeline:
    def __init__(self, config: ExperimentConfig, logger: logging.Logger) -> None: ...

    # ── Training mode ────────────────────────────────────────────────────
    def load_and_validate(self) -> pd.DataFrame:
        """
        Load CSV, verify schema, deduplicate, report counts.
        Raises: DataLoadError, SchemaValidationError
        """

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        One-hot encode categoricals, StandardScaler-normalise numericals.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
          where y is binary (0=normal, 1=attack).
        Fits and saves encoder + scaler artefacts.
        Raises: ConfigurationError (bad ratios), DataLoadError (too few normal samples)
        """

    def save_artefacts(self, run_dir: str) -> None:
        """Persist fitted OHE and scaler to run_dir as .pkl files."""

    # ── Inference mode ───────────────────────────────────────────────────
    def load_artefacts(self, artefact_dir: str) -> None:
        """
        Load fitted OHE and scaler from a previous run directory.
        Raises: ArtifactNotFoundError
        """

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply loaded encoder and scaler to new data.
        Unknown categorical values encoded as all-zero vectors.
        Raises: InferenceError (artefacts not loaded)
        """

    # ── Internal helpers ─────────────────────────────────────────────────
    def _validate_split_ratios(self) -> None: ...
    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def _report_counts(self, df: pd.DataFrame) -> None: ...
```

---

### 5. `model.py` — Autoencoder_Model

```python
class Autoencoder_Model:
    def __init__(
        self,
        arch_config: ArchitectureConfig,
        train_config: TrainingConfig,
        logger: logging.Logger,
    ) -> None: ...

    def build(self, input_dim: int) -> None:
        """
        Construct symmetric encoder-decoder graph.
        Architecture: Input → [Dense → BatchNorm → (Dropout)?]×n → Bottleneck
                             → [Dense → BatchNorm → (Dropout)?]×n → Output
        Raises: ConfigurationError (invalid arch params)
        """

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        run_dir: str,
    ) -> keras.callbacks.History:
        """
        Train on normal-traffic X_train, monitor on X_val.
        Applies early stopping, saves best weights.
        Raises: DataLoadError (empty X_train), ArtifactWriteError (unwritable path)
        """

    def reconstruct(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (X_reconstructed, per_sample_mse).
        Raises: InferenceError (model not built/compiled)
        """

    def predict(
        self, X: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (binary_predictions, mse_scores).
        binary_predictions[i] = 1 if mse_scores[i] >= threshold else 0.
        Raises: InferenceError (no threshold, wrong feature dim, empty batch)
        """

    def save_weights(self, path: str) -> None: ...
    def load_weights(self, path: str) -> None: ...

    # ── Internal ─────────────────────────────────────────────────────────
    def _build_lr_schedule(self) -> keras.optimizers.schedules.LearningRateSchedule | None: ...
    def _set_seeds(self) -> None: ...
```

**Architecture Detail:**

```
Input(input_dim)
  └─ Dense(encoder_layers[0], activation)
       └─ BatchNormalization()
            └─ Dropout(dropout_rate)          # only if dropout_rate > 0
                 └─ Dense(encoder_layers[1], activation)
                      └─ BatchNormalization()
                           └─ Dropout(dropout_rate)
                                └─ Dense(encoder_layers[2], activation)
                                     └─ BatchNormalization()
                                          └─ Dropout(dropout_rate)
                                               └─ Dense(bottleneck_dim, activation)   ← bottleneck
                                                    └─ Dense(encoder_layers[2], activation)
                                                         └─ BatchNormalization()
                                                              └─ Dropout(dropout_rate)
                                                                   └─ Dense(encoder_layers[1], activation)
                                                                        └─ BatchNormalization()
                                                                             └─ Dropout(dropout_rate)
                                                                                  └─ Dense(encoder_layers[0], activation)
                                                                                       └─ BatchNormalization()
                                                                                            └─ Dropout(dropout_rate)
                                                                                                 └─ Dense(input_dim, activation='linear')  ← output
Loss: MSE(input, output)
Optimiser: Adam(learning_rate)
```

---

### 6. `threshold.py` — Threshold_Estimator

```python
class Threshold_Estimator:
    def __init__(self, config: ThresholdConfig, logger: logging.Logger) -> None: ...

    def fit(self, mse_scores: np.ndarray) -> float:
        """
        Compute threshold at config.percentile of mse_scores.
        Logs threshold, percentile, mean, std.
        Returns the threshold value.
        """

    def save(self, run_dir: str) -> None:
        """
        Persist { threshold, percentile, mse_mean, mse_std } to
        <run_dir>/threshold.json
        """

    def load(self, artefact_dir: str) -> float:
        """
        Load threshold from <artefact_dir>/threshold.json.
        Raises: ArtifactLoadError (missing fields, invalid JSON)
        """

    @property
    def threshold(self) -> float:
        """Returns stored threshold; raises InferenceError if not yet computed/loaded."""
```

**Threshold JSON Schema:**
```json
{
  "threshold": 0.04217,
  "percentile": 95,
  "mse_mean": 0.01083,
  "mse_std": 0.00831
}
```

---

### 7. `evaluator.py` — Evaluator

```python
class Evaluator:
    def __init__(self, logger: logging.Logger) -> None: ...

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mse_scores: np.ndarray,
        threshold: float,
        run_dir: str,
    ) -> dict:
        """
        Compute all metrics and generate all plots.
        Raises: ValueError (length mismatch)
        Returns metrics dict (also written to metrics.json).
        """

    # ── Metric computation ───────────────────────────────────────────────
    def _compute_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict: ...

    def _compute_auc_roc(
        self, y_true: np.ndarray, mse_scores: np.ndarray
    ) -> float: ...

    # ── Plot generation ──────────────────────────────────────────────────
    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, run_dir: str
    ) -> None: ...

    def _plot_roc_curve(
        self, y_true: np.ndarray, mse_scores: np.ndarray, run_dir: str
    ) -> None: ...

    def _plot_re_histogram(
        self,
        mse_scores: np.ndarray,
        y_true: np.ndarray,
        threshold: float,
        run_dir: str,
    ) -> None: ...

    def _write_metrics_json(self, metrics: dict, run_dir: str) -> None: ...
```

**Metrics JSON Schema:**
```json
{
  "precision_normal":    0.973,
  "recall_normal":       0.961,
  "f1_normal":           0.967,
  "precision_anomalous": 0.942,
  "recall_anomalous":    0.958,
  "f1_anomalous":        0.950,
  "accuracy":            0.962,
  "auc_roc":             0.991,
  "training_time_secs":  42.17
}
```

---

### 8. `experiment_tracker.py` — Experiment_Tracker

```python
class Experiment_Tracker:
    def __init__(self, yaml_path: str) -> None: ...

    def setup(self) -> ExperimentConfig:
        """
        Load YAML config, create timestamped run dir, set up logger,
        copy config file, log git hash.
        Raises: ConfigurationError (missing YAML keys)
        """

    def run_training(self) -> None:
        """Orchestrates: load → preprocess → build → train → threshold → evaluate → save."""

    def run_inference(self, csv_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Orchestrates inference pipeline.
        Loads artefacts, preprocesses, runs predict().
        Raises: ArtifactNotFoundError (any critical artefact missing)
        """

    def _create_run_dir(self) -> str:
        """Returns path <output.artefact_dir>/YYYYMMDD_HHMMSS/"""

    def _log_git_hash(self) -> None: ...
    def _record_timing(self, elapsed_secs: float) -> None: ...
```

---

## Data Models

### Feature Vector Dimensionality

| Dataset   | Raw features | Categorical cols | After OHE (approx.) | After StandardScaler |
|-----------|-------------|-------------------|----------------------|----------------------|
| NSL-KDD   | 41          | 3                 | ~122                 | ~122 (same dim)      |
| CICIDS2017| 80          | 0–2               | ~80                  | ~80                  |
| UNSW-NB15 | 49          | 3–4               | ~196                 | ~196                 |

> The exact post-OHE dimensionality is determined at fit-time and stored implicitly in the serialised encoder artefact. The Autoencoder `input_dim` is set to this value when `build()` is called.

### NSL-KDD Label Mapping

```python
# Binary label for anomaly detection
BINARY_LABEL_MAP = {
    "normal": 0,
    # All attack categories → 1
    "dos": 1, "probe": 1, "r2l": 1, "u2r": 1,
    # raw subcategory values also map to 1
}
```

### Split Invariants

```
len(X_train) + len(X_val) + len(X_test) == len(df_normal) + len(df_attack)
X_train contains only y==0 samples
X_val   contains only y==0 samples
X_test  contains both y==0 and y==1 samples
```

### Artefact Directory Layout

```
runs/
└── 20241215_143022/
    ├── config.yaml              # byte-for-byte copy of input YAML
    ├── model_weights.keras      # Keras SavedModel / .keras format
    ├── scaler.pkl               # sklearn StandardScaler (joblib)
    ├── encoder.pkl              # sklearn OneHotEncoder (joblib)
    ├── threshold.json           # { threshold, percentile, mse_mean, mse_std }
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── re_histogram.png
    ├── metrics.json             # all scalar metrics + training_time_secs
    └── experiment.log           # full structured log for this run
```

### Reference YAML Configuration

```yaml
dataset:
  path: "data/KDDTrain+.csv"
  schema: "nsl_kdd"

splits:
  train_ratio: 0.70
  val_ratio:   0.10
  test_ratio:  0.20
  random_seed: 42

architecture:
  encoder_layers: [128, 64, 32]
  bottleneck_dim: 16
  activation: "relu"
  dropout_rate: 0.2

training:
  learning_rate: 0.001
  batch_size:    256
  max_epochs:    100
  patience:      10
  lr_schedule:   "none"
  random_seed:   42

threshold:
  percentile: 95

output:
  artefact_dir: "runs"
```


---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

---

### Property 1: Schema Column-Count Validation

*For any* dataset file and schema selection, the Data_Pipeline SHALL accept the file if and only if the number of feature columns exactly equals the schema's declared feature count (41 for NSL-KDD, 80 for CICIDS2017, 49 for UNSW-NB15), and SHALL raise a `SchemaValidationError` naming all missing columns otherwise.

**Validates: Requirements 1.1, 1.3**

---

### Property 2: Missing File Raises DataLoadError with Path

*For any* file path that does not exist or is unreadable, calling `Data_Pipeline.load_and_validate()` SHALL raise a `DataLoadError` whose message contains the offending path string.

**Validates: Requirements 1.2**

---

### Property 3: Deduplication Idempotence

*For any* dataframe with an arbitrary number of injected duplicate rows, after `_deduplicate()` the result SHALL contain no duplicate rows, and applying `_deduplicate()` a second time SHALL produce a result of identical shape (idempotence).

**Validates: Requirements 1.5**

---

### Property 4: Count Invariant — Total Equals Normal Plus Attack

*For any* successfully loaded dataframe, the reported total record count SHALL equal the sum of the reported normal-traffic count and the reported attack-traffic count.

**Validates: Requirements 1.4**

---

### Property 5: OHE and Feature Dimensionality Consistency

*For any* valid dataset, after `fit_transform()` completes, the training split, validation split, and test split SHALL all have identical second-dimension size (feature vector length), and that dimension SHALL be strictly greater than the raw feature count (due to one-hot expansion of categorical columns).

**Validates: Requirements 2.1, 2.5**

---

### Property 6: StandardScaler Normalisation Invariant

*For any* training feature matrix with at least 2 rows, after `fit_transform()` the per-column mean of the training split's continuous numerical features SHALL be within ±0.01 of 0.0, and the per-column standard deviation SHALL be within ±0.01 of 1.0.

**Validates: Requirements 2.2**

---

### Property 7: Unknown Categorical Value Encoded as All-Zero Vector

*For any* inference-mode call to `transform()` that includes a categorical value not seen during fitting, all one-hot indicator columns corresponding to that categorical feature SHALL be 0.0 in the output row for that record.

**Validates: Requirements 2.3**

---

### Property 8: Split-Ratio Validity and Size Proportionality

*For any* valid ratio triple `(train_ratio, val_ratio, test_ratio)` where each ratio ∈ (0.0, 1.0) and their sum equals 1.0, the resulting splits SHALL have sizes proportional to the given ratios (within ±1 row due to rounding). For any invalid ratio triple (any ratio ≤ 0.0 or ≥ 1.0, or sum ≠ 1.0), `fit_transform()` SHALL raise a `ConfigurationError` naming the invalid ratio.

**Validates: Requirements 2.4**

---

### Property 9: Autoencoder Symmetric Architecture with BatchNorm and Dropout

*For any* valid `ArchitectureConfig` with encoder_layers `[l₁, l₂, l₃]`, the built model SHALL have a symmetric decoder with layers `[l₃, l₂, l₁]` mirroring the encoder, every hidden layer SHALL be followed by a `BatchNormalization` layer, and when `dropout_rate > 0` every `BatchNormalization` layer SHALL be followed by a `Dropout` layer; when `dropout_rate = 0` no `Dropout` layers SHALL appear.

**Validates: Requirements 3.1, 3.3, 3.4**

---

### Property 10: ConfigurationError for Out-of-Range Parameters

*For any* `ArchitectureConfig` or `TrainingConfig` where any single parameter falls outside its valid range (encoder layer size < 1, bottleneck_dim ≥ min(encoder_layers), dropout_rate ∉ [0.0, 1.0), learning_rate ≤ 0, batch_size < 1, max_epochs < 1, patience < 1), `build()` or `fit()` SHALL raise a `ConfigurationError` whose message names the invalid parameter.

**Validates: Requirements 3.2, 4.2**

---

### Property 11: reconstruct(X) Output Shape Invariant

*For any* input matrix `X` with shape `(n, d)` passed to a built and compiled `Autoencoder_Model`, `reconstruct(X)` SHALL return a tuple `(X_hat, mse)` where `X_hat.shape == (n, d)` and `mse.shape == (n,)` with all MSE values ≥ 0.

**Validates: Requirements 3.6**

---

### Property 12: predict(X) Binary Threshold Classification Correctness

*For any* input matrix `X` with the correct feature dimensionality, and any loaded threshold `τ`, `predict(X)` SHALL return `(predictions, mse_scores)` such that `predictions[i] == 1` if and only if `mse_scores[i] >= τ`, for every row `i`. If the input feature count differs from the training feature count, `predict(X)` SHALL raise an `InferenceError` stating both the expected and actual counts.

**Validates: Requirements 6.1, 6.2, 6.5**

---

### Property 13: Threshold Round-Trip — Compute, Save, Load

*For any* non-empty MSE vector and percentile `p` ∈ [1, 99], `Threshold_Estimator.fit()` SHALL return a threshold equal to `np.percentile(mse_scores, p)`, `save()` SHALL write a JSON file containing all four required fields (`threshold`, `percentile`, `mse_mean`, `mse_std`) with values matching the computed statistics, and `load()` from that file SHALL return the same threshold value. If any required field is missing or the file is not valid JSON, `load()` SHALL raise an `ArtifactLoadError` identifying the problematic fields.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

---

### Property 14: Evaluator Metrics Validity and JSON Completeness

*For any* binary ground-truth label vector `y_true` and corresponding MSE scores `mse_scores` of equal length, `Evaluator.evaluate()` SHALL produce values for precision, recall, F1-score (for both classes), accuracy, and AUC-ROC where each value is in `[0.0, 1.0]`, and the resulting `metrics.json` SHALL contain all eight required keys. For a perfectly separable `mse_scores` vector (all attack scores strictly above all normal scores), the computed AUC-ROC SHALL equal 1.0.

**Validates: Requirements 7.1, 7.2, 7.6**

---

### Property 15: ArtifactNotFoundError Lists All Missing Files

*For any* subset of the four inference-critical artefact files (model weights, scaler, encoder, threshold JSON) that is absent from the artefact directory, `Experiment_Tracker.run_inference()` SHALL raise an `ArtifactNotFoundError` that lists all and only the absent file names, and no component SHALL enter a partially loaded state.

**Validates: Requirements 9.4**

---

### Property 16: Artefact Completeness After Training

*For any* valid training run on synthetic normal-traffic data, after `run_training()` completes, the run subdirectory SHALL contain all nine required artefact files: `model_weights.keras`, `scaler.pkl`, `encoder.pkl`, `threshold.json`, `confusion_matrix.png`, `roc_curve.png`, `re_histogram.png`, `metrics.json`, and `config.yaml`.

**Validates: Requirements 9.2**

---

### Property 17: Config File Byte-for-Byte Copy

*For any* YAML configuration file passed to `Experiment_Tracker.setup()`, the file copied into the run artefact directory SHALL be byte-for-byte identical to the original input file (same content hash, same filename).

**Validates: Requirements 8.2**

---

### Property 18: Missing YAML Key Raises ConfigurationError Naming the Key

*For any* YAML configuration file with any one of the six required top-level keys (`dataset`, `splits`, `architecture`, `training`, `threshold`, `output`) omitted, `load_config()` SHALL raise a `ConfigurationError` whose message includes the name of the missing key before any data loading or component initialisation occurs.

**Validates: Requirements 8.1, 8.5**

---

## Error Handling

### Exception Hierarchy Design

All custom exceptions extend `NetworkAnomalyBaseError` to allow catch-all handling at the top level while still supporting fine-grained handling per exception type. The hierarchy is flat (one level deep) to avoid import complexity.

```python
NetworkAnomalyBaseError
├── DataLoadError          # I/O failures, empty files, insufficient normal samples
├── SchemaValidationError  # Missing required columns in loaded data
├── ConfigurationError     # Bad YAML keys or out-of-range parameters
├── InferenceError         # Model/threshold not ready, shape mismatch, empty batch
├── ArtifactLoadError      # Corrupt or incomplete saved artefact
├── ArtifactNotFoundError  # Expected artefact file(s) absent from directory
└── ArtifactWriteError     # Unwritable file path for saving artefact
```

### Error Propagation Pattern

```
Component method
  ├── raises specific exception immediately on detection (no silent suppression)
  ├── Experiment_Tracker catches at phase boundaries
  │     └── logs (ERROR: type + message + traceback)
  │         then re-raises (preserves original stack trace via `raise` not `raise e`)
  └── main.py catches NetworkAnomalyBaseError at top level
        └── exits with non-zero code and final error message to stderr
```

### Error Message Standards

Every exception message must include:
- **DataLoadError**: file path; or description of why data is insufficient
- **SchemaValidationError**: list of all missing column names
- **ConfigurationError**: name of invalid/missing key and valid range or expected type
- **InferenceError**: which artefact is missing OR expected vs actual feature count
- **ArtifactLoadError**: which fields are missing or which file is unparseable
- **ArtifactNotFoundError**: list of all missing file names
- **ArtifactWriteError**: the unwritable path

### Partial-State Prevention (Requirement 9.4)

Inference artefact loading follows an **all-or-nothing** pattern:
1. `Experiment_Tracker` checks all four artefact files exist before loading any
2. If any are missing, `ArtifactNotFoundError` is raised with the full missing-file list
3. No component's `load_artefacts()` / `load_weights()` / `load()` method is called until all files are confirmed present

---

## Testing Strategy

### Dual Testing Approach

The project uses a **two-layer** testing strategy: unit/property tests for logical correctness and integration tests for end-to-end behaviour.

```
tests/
├── property/     ← Hypothesis-powered; 100+ iterations each; verifies Properties 1–18
├── unit/         ← Concrete examples, edge cases, error conditions
└── integration/  ← Full pipeline runs on small synthetic data; timing assertions
```

### Property-Based Testing

**Library**: [Hypothesis](https://hypothesis.readthedocs.io/) (`pip install hypothesis`)

Each of the 18 correctness properties is implemented as a **single** Hypothesis property test, tagged with the property number:

```python
# Tag format:
# @settings(max_examples=100)
# Feature: network-anomaly-autoencoder, Property N: <property text>

from hypothesis import given, settings
from hypothesis import strategies as st

@given(st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=10))
@settings(max_examples=200)
def test_threshold_round_trip(mse_values):
    # Feature: network-anomaly-autoencoder, Property 13: Threshold round-trip
    mse = np.array(mse_values)
    estimator = Threshold_Estimator(ThresholdConfig(percentile=95), logger)
    threshold = estimator.fit(mse)
    assert abs(threshold - np.percentile(mse, 95)) < 1e-9
    # ... save/load round-trip
```

**Minimum iterations**: 100 per property (default); 200 for numerical properties to stress floating-point edge cases.

**Hypothesis strategies** to use:
| Property | Strategy |
|----------|----------|
| Schema validation (P1) | `st.integers(min_value=1, max_value=200)` for column counts |
| File paths (P2) | `st.text(alphabet=st.characters(blacklist_categories=('Cc',)))` |
| Ratio triples (P8) | Custom: `st.floats(0.01, 0.98)` composed to sum to 1 |
| MSE vectors (P13) | `st.lists(st.floats(min_value=0.0, max_value=100.0, allow_nan=False))` |
| Binary labels (P14) | `st.lists(st.integers(min_value=0, max_value=1), min_size=10)` |
| Arch configs (P9) | `st.builds(ArchitectureConfig, ...)` with valid ranges |
| Feature matrices (P11, P12) | `st.arrays(np.float32, shape=st.tuples(st.integers(1,500), st.just(input_dim)))` |

### Unit Tests

Focus areas:
- **Error-condition edge cases**: empty datasets (1.6, 2.6, 4.6), zero batch inference (6.6), wrong feature count (6.5), unwritable paths (4.7), AUC < 0.80 warning (7.7)
- **Specific behavioural examples**: MSE loss function verification (3.5), log format verification (10.4), git hash logging (8.3)
- **Plot file creation**: confusion matrix, ROC curve, RE histogram files (7.3, 7.4, 7.5)
- **Timestamp directory format**: YYYYMMDD_HHMMSS pattern (9.1)

### Integration Tests

| Test | Coverage | Assertion |
|------|----------|-----------|
| End-to-end training on synthetic data | All phases | All 9 artefacts present; metrics.json valid |
| Inference from artefact dir | 9.3 | Predictions shape matches input; no exceptions |
| Early stopping | 4.3 | Training stops before max_epochs on overfit data |
| CPU inference timing | 6.4 | 10,000 samples predicted in < 5 seconds |

### Test Data Strategy

- **Synthetic NSL-KDD-like data**: generated via `numpy.random` with configurable proportions of normal/attack rows; no external dataset download required for tests
- **Minimal sizes**: 200–500 rows for training tests; 10,000 rows for the timing integration test
- **Fixtures**: `pytest` fixtures in `conftest.py` provide reusable synthetic data arrays and temp directories

### Coverage Goals

| Layer | Target |
|-------|--------|
| Property tests | All 18 correctness properties covered |
| Unit tests | ≥ 90% line coverage on all non-UI code |
| Integration tests | 1 full training run + 1 inference run per CI run |
