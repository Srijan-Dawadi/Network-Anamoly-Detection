"""
exceptions.py
=============
Custom exception hierarchy for the network-anomaly-autoencoder project.

All exceptions extend NetworkAnomalyBaseError so callers can choose between
fine-grained handling (catch a specific subclass) and broad handling
(catch NetworkAnomalyBaseError).
"""


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
