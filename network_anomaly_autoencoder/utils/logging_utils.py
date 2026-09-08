"""
Logging utility factory for network-anomaly-autoencoder.

Provides a `get_logger` factory that creates (or retrieves) a named logger
with a consistent format and optional file output.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s"


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Return a named logger configured with a StreamHandler (stdout) and,
    optionally, a FileHandler writing to *log_file*.

    If the logger already has handlers attached (i.e. this function was called
    previously with the same *name*), no duplicate handlers are added.

    Parameters
    ----------
    name : str
        Logger name — typically the module's ``__name__`` or an experiment name.
    log_file : str, optional
        Filesystem path of the log file.  When provided, log output is written
        to that file **in addition to** the console stream.

    Returns
    -------
    logging.Logger
        Configured logger at INFO level.
    """
    logger = logging.getLogger(name)

    # Set level only when first configuring this logger to avoid overriding
    # a caller who may have already set a stricter level.
    if not logger.handlers:
        logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT)

    # --- StreamHandler (stdout) -------------------------------------------
    # Guard: only add if no StreamHandler to the same stream is already present.
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) is sys.stdout
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # --- FileHandler (optional) -------------------------------------------
    if log_file is not None:
        import os

        abs_log_file = os.path.abspath(log_file)
        # Guard: only add if no FileHandler pointing to the same path already exists.
        has_file_handler = any(
            isinstance(h, logging.FileHandler)
            and os.path.abspath(getattr(h, "baseFilename", "")) == abs_log_file
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
