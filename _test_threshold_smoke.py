import logging
import json
import os
import tempfile
import numpy as np

from network_anomaly_autoencoder.threshold import Threshold_Estimator
from network_anomaly_autoencoder.config import ThresholdConfig
from network_anomaly_autoencoder.exceptions import InferenceError, ArtifactLoadError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke")

cfg = ThresholdConfig(percentile=95)
est = Threshold_Estimator(cfg, logger)

# threshold property before fit should raise InferenceError
try:
    _ = est.threshold
    raise AssertionError("FAIL: expected InferenceError")
except InferenceError as e:
    print(f"OK: InferenceError raised before fit: {e}")

# fit
rng = np.random.default_rng(42)
mse = rng.uniform(0.001, 0.1, 1000)
t = est.fit(mse)
expected = float(np.percentile(mse, 95))
assert abs(t - expected) < 1e-12, f"threshold mismatch: {t} vs {expected}"
print(f"OK: fit returned {t:.6f} (expected {expected:.6f})")

# threshold property after fit
assert est.threshold == t
print("OK: threshold property works after fit")

# save / load round-trip
with tempfile.TemporaryDirectory() as tmp:
    est.save(tmp)
    path = os.path.join(tmp, "threshold.json")
    with open(path) as f:
        data = json.load(f)
    assert set(data.keys()) == {"threshold", "percentile", "mse_mean", "mse_std"}, \
        f"unexpected keys: {set(data.keys())}"
    print(f"OK: saved keys = {sorted(data.keys())}")

    est2 = Threshold_Estimator(cfg, logger)
    t2 = est2.load(tmp)
    assert abs(t2 - t) < 1e-12, f"round-trip mismatch: {t2} vs {t}"
    print(f"OK: load round-trip returned {t2:.6f}")

    # corrupt JSON → ArtifactLoadError
    with open(path, "w") as f:
        f.write("{bad json")
    try:
        est2.load(tmp)
        raise AssertionError("FAIL: expected ArtifactLoadError on bad JSON")
    except ArtifactLoadError as e:
        print(f"OK: ArtifactLoadError on bad JSON: {e}")

    # missing fields → ArtifactLoadError
    with open(path, "w") as f:
        json.dump({"threshold": 0.05, "percentile": 95}, f)
    try:
        est2.load(tmp)
        raise AssertionError("FAIL: expected ArtifactLoadError on missing fields")
    except ArtifactLoadError as e:
        print(f"OK: ArtifactLoadError on missing fields: {e}")

    # unparseable field → ArtifactLoadError
    with open(path, "w") as f:
        json.dump({"threshold": "not-a-number", "percentile": 95, "mse_mean": 0.01, "mse_std": 0.005}, f)
    try:
        est2.load(tmp)
        raise AssertionError("FAIL: expected ArtifactLoadError on unparseable field")
    except ArtifactLoadError as e:
        print(f"OK: ArtifactLoadError on unparseable field: {e}")

print("\nAll smoke checks passed.")
