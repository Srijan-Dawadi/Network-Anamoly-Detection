# Requirements Document

## Introduction

This document specifies requirements for an advanced network anomaly detection system that uses a deep Autoencoder neural network to identify anomalous traffic patterns in network flow data. The system ingests pre-processed network flow records, trains an Autoencoder on normal traffic to learn a compressed representation, and flags records whose reconstruction error exceeds a learned threshold as anomalies. The primary dataset target is **NSL-KDD** — a cleaned, de-duplicated derivative of KDD Cup 99 that is the standard benchmark for network intrusion detection research — though the pipeline is designed to support CICIDS2017 and UNSW-NB15 with minimal adapter changes.

The project is intended as an advanced deep learning practice exercise and should reflect production-quality design: modular code, reproducible experiments, robust evaluation, and clear observability.

---

## Glossary

- **Autoencoder**: A neural network trained to compress an input into a latent representation and reconstruct the original input. Anomalies produce high reconstruction error.
- **Reconstruction Error (RE)**: The mean squared error (MSE) between the original input and the Autoencoder's reconstruction.
- **Anomaly Threshold**: The RE value above which a sample is classified as anomalous. Derived from the 95th–99th percentile of RE on the normal training set.
- **NSL-KDD**: The benchmark network intrusion dataset derived from KDD Cup 99, containing 41 features per connection record and multi-class attack labels.
- **CICIDS2017**: A modern network intrusion dataset containing 80 features derived from bidirectional network flows.
- **UNSW-NB15**: A modern dataset with 49 features covering nine attack categories generated in a hybrid real/simulated network environment.
- **Normal Traffic**: Network flow records labelled as benign/normal, used exclusively to train the Autoencoder.
- **Anomalous Traffic**: Network flow records labelled as any attack category; expected to produce high RE.
- **Data_Pipeline**: The module responsible for loading, validating, encoding, and scaling raw dataset files.
- **Autoencoder_Model**: The Keras/TensorFlow deep Autoencoder network definition, training, and inference logic.
- **Threshold_Estimator**: The component that computes and persists the anomaly threshold from training-set reconstruction errors.
- **Evaluator**: The component that computes classification metrics (AUC-ROC, precision, recall, F1, confusion matrix) from predictions and ground-truth labels.
- **Experiment_Tracker**: The component that logs hyperparameters, metrics, and artefacts for each training run.
- **Artefact**: A saved file produced by training — model weights, scaler, threshold value, evaluation plots.

---

## Requirements

### Requirement 1: Dataset Loading and Validation

**User Story:** As a researcher, I want the system to load and validate raw NSL-KDD CSV files, so that I can be confident the data fed into training is structurally correct.

#### Acceptance Criteria

1. WHEN a dataset file path is provided, THE Data_Pipeline SHALL load the file and verify it contains the expected number of feature columns for the selected dataset schema (41 for NSL-KDD, 80 for CICIDS2017, 49 for UNSW-NB15).
2. IF a dataset file is missing or unreadable, THEN THE Data_Pipeline SHALL raise a descriptive `DataLoadError` with the offending file path.
3. IF required columns are absent from the loaded file, THEN THE Data_Pipeline SHALL raise a `SchemaValidationError` listing all missing columns.
4. WHEN a dataset file is successfully loaded, THE Data_Pipeline SHALL report the total record count, the count of records whose label column equals the dataset's normal-traffic label value, and the count of all remaining records as the attack record count.
5. WHEN duplicate rows are detected (defined as rows where all feature columns and the label column are identical), THE Data_Pipeline SHALL log the duplicate count at INFO level and remove them before returning the dataset.
6. IF the dataset file is readable but contains zero records after header parsing, THEN THE Data_Pipeline SHALL raise a `DataLoadError` indicating the file is empty.

---

### Requirement 2: Feature Engineering and Preprocessing

**User Story:** As a researcher, I want raw categorical and numerical features encoded and normalised, so that the Autoencoder receives well-conditioned numerical input.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL one-hot encode all categorical features (e.g., `protocol_type`, `service`, `flag` in NSL-KDD), leaving all non-categorical features unchanged, and store the fitted encoder as a serialised artefact.
2. THE Data_Pipeline SHALL apply `StandardScaler` normalisation to all continuous numerical features and store the fitted scaler as a serialised artefact.
3. WHEN the pipeline is run in inference mode, THE Data_Pipeline SHALL apply the previously fitted encoder and scaler rather than re-fitting on new data. IF a categorical value is encountered during inference that was not present during training, THE Data_Pipeline SHALL encode it as an all-zero vector for that feature's one-hot columns.
4. THE Data_Pipeline SHALL split the dataset into a training set (normal traffic only), a validation set (normal traffic only), and a test set (normal + anomalous traffic) using configurable split ratios where each ratio is in the range (0.0, 1.0) exclusive and all three ratios sum to 1.0, with a fixed random seed. IF the ratios do not satisfy these constraints, THE Data_Pipeline SHALL raise a `ConfigurationError` naming the invalid ratio.
5. THE Data_Pipeline SHALL produce a numeric feature vector of identical dimensionality for every input record across training, validation, and test splits.
6. IF the normal-traffic subset contains fewer than 2 records after splitting, THEN THE Data_Pipeline SHALL raise a `DataLoadError` indicating insufficient normal-traffic samples for training.

---

### Requirement 3: Autoencoder Architecture

**User Story:** As a deep learning practitioner, I want a configurable deep Autoencoder with bottleneck regularisation, so that I can experiment with architecture depth and latent space size.

#### Acceptance Criteria

1. THE Autoencoder_Model SHALL implement a symmetric encoder-decoder architecture with at least three hidden layers on each side.
2. THE Autoencoder_Model SHALL support configurable layer sizes (each ≥ 1 neuron), bottleneck dimension (≥ 1 and less than the smallest encoder hidden layer size), activation function, and dropout rate (in range [0.0, 1.0)) through a configuration object. IF any configuration value falls outside these valid ranges, THE Autoencoder_Model SHALL raise a `ConfigurationError` naming the invalid parameter before building the network.
3. THE Autoencoder_Model SHALL apply Batch Normalisation after each hidden layer.
4. WHERE the dropout rate configuration value is greater than zero, THE Autoencoder_Model SHALL apply Dropout regularisation after each Batch Normalisation layer.
5. THE Autoencoder_Model SHALL use Mean Squared Error as the reconstruction loss function during training.
6. THE Autoencoder_Model SHALL expose a `reconstruct(X)` method that returns a tuple of (reconstructed feature matrix with the same shape as `X`, per-sample MSE vector of length equal to the number of rows in `X`). IF `reconstruct(X)` is called before the model has been built and compiled, THE Autoencoder_Model SHALL raise an `InferenceError` with a message indicating the model is not yet initialised.

---

### Requirement 4: Model Training

**User Story:** As a researcher, I want reproducible, monitored training with early stopping, so that I can halt training at the optimal validation loss without overfitting.

#### Acceptance Criteria

1. WHEN training is initiated, THE Autoencoder_Model SHALL train exclusively on normal-traffic samples.
2. THE Autoencoder_Model SHALL support configurable hyperparameters: learning rate (> 0.0), batch size (≥ 1 integer), maximum epochs (≥ 1 integer), early-stopping patience (≥ 1 integer), and learning-rate schedule (one of: `none`, `step_decay`, `cosine_annealing`) through a configuration object.
3. WHEN validation loss does not strictly decrease below the best recorded value for a consecutive number of epochs equal to the early-stopping patience, THE Autoencoder_Model SHALL restore the best weights and stop training.
4. THE Autoencoder_Model SHALL save the best model weights to a file path specified in the configuration after training completes.
5. WHEN a random seed is set in the configuration, THE Autoencoder_Model SHALL initialise all random number generators (Python `random`, NumPy, TensorFlow) with that seed so that two training runs with identical configuration produce identical weight artefacts.
6. IF the training split contains zero normal-traffic samples, THEN THE Autoencoder_Model SHALL raise a `DataLoadError` before training begins, indicating no normal-traffic samples are available.
7. IF the file path specified for saving model weights is not writable, THEN THE Autoencoder_Model SHALL raise an `ArtifactWriteError` naming the unwritable path before training begins.

---

### Requirement 5: Anomaly Threshold Estimation

**User Story:** As a researcher, I want an automatically computed anomaly threshold derived from training-set reconstruction errors, so that I do not need to manually tune a cutoff value.

#### Acceptance Criteria

1. WHEN training completes, THE Threshold_Estimator SHALL compute per-sample MSE on the normal training set using the trained Autoencoder_Model.
2. THE Threshold_Estimator SHALL set the anomaly threshold at a configurable percentile in the range [1, 99] inclusive (default 95) of the training-set MSE distribution.
3. THE Threshold_Estimator SHALL persist the computed threshold value to a JSON artefact file in the run artefact directory. The JSON file SHALL contain at minimum the following fields: `threshold`, `percentile`, `mse_mean`, and `mse_std`.
4. WHEN a previously saved threshold artefact is loaded, THE Threshold_Estimator SHALL return the stored value without recomputation. IF the artefact file is missing any required field or is not valid JSON, THE Threshold_Estimator SHALL raise an `ArtifactLoadError` identifying the missing or unparseable fields.
5. IF a previously saved threshold artefact file is corrupt or unparseable, THEN THE Threshold_Estimator SHALL raise an `ArtifactLoadError` identifying the missing or unparseable fields.
6. WHEN the threshold is computed, THE Threshold_Estimator SHALL log the threshold value, the percentile used, and the mean and standard deviation of the training-set MSE distribution at INFO level.

---

### Requirement 6: Anomaly Detection Inference

**User Story:** As an operator, I want to classify new network flow records as normal or anomalous in batch, so that I can identify threats in captured traffic.

#### Acceptance Criteria

1. WHEN a batch of preprocessed feature vectors is provided to `predict(X)`, THE Autoencoder_Model SHALL compute per-sample MSE and return a binary prediction vector where samples with MSE strictly greater than or equal to the anomaly threshold are labelled 1 (anomalous) and samples with MSE below the threshold are labelled 0 (normal).
2. WHEN `predict(X)` is called, THE Autoencoder_Model SHALL return both the binary prediction vector and the raw per-sample MSE vector as a single tuple `(predictions, mse_scores)` in a single call.
3. WHEN inference is requested before a threshold artefact has been loaded or computed, THE Autoencoder_Model SHALL raise an `InferenceError` whose message explicitly names which missing artefact (model weights or threshold) prevents inference.
4. WHEN `predict(X)` is called on a system with no GPU acceleration available, THE Autoencoder_Model SHALL complete inference on a batch of 10,000 samples within 5 seconds of wall-clock time.
5. IF the number of features in input `X` does not match the dimensionality used during training, THEN THE Autoencoder_Model SHALL raise an `InferenceError` stating the expected and actual feature counts.
6. IF `predict(X)` is called with an input containing zero samples, THEN THE Autoencoder_Model SHALL raise an `InferenceError` indicating the input batch is empty.

---

### Requirement 7: Evaluation and Metrics

**User Story:** As a researcher, I want comprehensive classification metrics and visualisations, so that I can assess model performance and compare experiments.

#### Acceptance Criteria

1. WHEN ground-truth labels (0 = normal, 1 = anomalous), binary predictions, and raw MSE scores are provided, THE Evaluator SHALL compute precision, recall, F1-score, and accuracy for both the anomalous class (label 1) and the normal class (label 0). IF the lengths of labels, predictions, or MSE scores do not match, THE Evaluator SHALL raise a `ValueError` identifying the mismatched lengths before computing any metric.
2. THE Evaluator SHALL compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) using the raw MSE scores as the ranking signal.
3. THE Evaluator SHALL generate and save a confusion matrix heatmap as a PNG artefact.
4. THE Evaluator SHALL generate and save an ROC curve plot as a PNG artefact.
5. THE Evaluator SHALL generate and save a histogram of reconstruction errors for normal and anomalous samples on the test set, with the anomaly threshold overlaid as a vertical line.
6. THE Evaluator SHALL write all scalar metrics — precision (normal and anomalous), recall (normal and anomalous), F1-score (normal and anomalous), accuracy, and AUC-ROC — to a JSON report file alongside the PNG artefacts.
7. IF the AUC-ROC score on the test set is below 0.80, THEN THE Evaluator SHALL emit a WARNING-level log message indicating that the model may be underperforming.

---

### Requirement 8: Experiment Configuration and Reproducibility

**User Story:** As a researcher, I want all experiment settings in a single YAML configuration file, so that I can reproduce any run by re-supplying the same file.

#### Acceptance Criteria

1. THE Experiment_Tracker SHALL read all runtime parameters from a YAML configuration file. The required top-level keys are: `dataset` (containing dataset path and schema name), `splits` (containing train/validation/test ratios and random seed), `architecture` (containing layer sizes, bottleneck dimension, activation, dropout rate), `training` (containing learning rate, batch size, max epochs, patience, and LR schedule), `threshold` (containing percentile), and `output` (containing artefact output directory path).
2. WHEN an experiment is run, THE Experiment_Tracker SHALL copy the configuration file byte-for-byte, preserving the original filename, into the artefact output directory.
3. IF the project directory is inside a Git repository, THEN THE Experiment_Tracker SHALL log the current Git commit hash at INFO level at experiment start. IF the project is not inside a Git repository, THEN THE Experiment_Tracker SHALL log the string `"git-unavailable"` in place of the commit hash.
4. WHEN training completes, THE Experiment_Tracker SHALL record the wall-clock training time in seconds with two decimal places of precision and write it to the metrics JSON report file.
5. IF a required configuration key is missing from the YAML file, THEN THE Experiment_Tracker SHALL raise a `ConfigurationError` naming the missing key before any data is loaded.

---

### Requirement 9: Artefact Persistence and Loading

**User Story:** As an operator, I want all model artefacts saved in a versioned output directory, so that I can reload any past experiment for inference without retraining.

#### Acceptance Criteria

1. WHEN a training run is initiated, THE Experiment_Tracker SHALL create a timestamped subdirectory within the configured output directory using the format `YYYYMMDD_HHMMSS`.
2. WHEN a training run completes, THE Experiment_Tracker SHALL save the following artefacts into the run subdirectory: model weights file, fitted scaler, fitted encoder, threshold JSON, confusion matrix heatmap PNG, ROC curve PNG, reconstruction error histogram PNG, and metrics JSON report.
3. WHEN an inference run specifies an artefact directory path, THE Data_Pipeline SHALL load the fitted scaler and fitted encoder from that directory, THE Autoencoder_Model SHALL load the model weights file from that directory, and THE Threshold_Estimator SHALL load the threshold JSON from that directory.
4. IF any of the four inference-critical artefact files (model weights, scaler, encoder, threshold JSON) is absent from the specified directory during inference loading, THEN THE Experiment_Tracker SHALL raise an `ArtifactNotFoundError` listing all missing file names, and no component SHALL enter a partially loaded state.

---

### Requirement 10: Logging and Observability

**User Story:** As a developer, I want structured, levelled log output at every major processing step, so that I can diagnose failures and monitor long-running training jobs.

#### Acceptance Criteria

1. THE Experiment_Tracker SHALL emit structured log messages at INFO level at the start and end of each major phase: data loading, preprocessing, model initialisation, training, threshold estimation, inference, and evaluation.
2. WHEN an unhandled exception occurs in any component, THE Experiment_Tracker SHALL log the exception type, message, and stack trace at ERROR level before re-raising.
3. THE Autoencoder_Model SHALL log training loss and validation loss at the end of each epoch at DEBUG level.
4. WHERE a console handler is configured, THE Experiment_Tracker SHALL format log output with timestamps, log level, module name, and message.
5. THE Experiment_Tracker SHALL write all log output to a log file inside the run artefact directory in addition to any console output.
