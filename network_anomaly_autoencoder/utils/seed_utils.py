"""
utils/seed_utils.py
===================
Global random seed utility for reproducible experiments.
"""

import random
import numpy as np
import tensorflow as tf


def set_global_seeds(seed: int) -> None:
    """Set all random number generators for reproducibility.

    Sets Python's built-in ``random``, NumPy, and TensorFlow seeds
    in a single call so that two runs with the same seed produce
    identical results.

    Parameters
    ----------
    seed : int
        The integer seed value to apply to all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
