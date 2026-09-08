"""
main.py
=======
Command-line entry point for the network-anomaly autoencoder.

Usage:
    python main.py train --config configs/nsl_kdd_default.yaml
    python main.py infer  --config configs/nsl_kdd_default.yaml --data data/new_traffic.csv
"""

import argparse
import os
import sys

from network_anomaly_autoencoder.config import load_config
from network_anomaly_autoencoder.exceptions import NetworkAnomalyBaseError
from network_anomaly_autoencoder.experiment_tracker import Experiment_Tracker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="network-anomaly-autoencoder",
        description="Detect anomalous network traffic with a deep autoencoder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training experiment.")
    train_parser.add_argument(
        "--config", required=True, help="Path to the YAML experiment configuration."
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference on new data.")
    infer_parser.add_argument(
        "--config", required=True, help="Path to the YAML experiment configuration."
    )
    infer_parser.add_argument(
        "--data", required=True, help="Path to the CSV file to classify."
    )
    infer_parser.add_argument(
        "--run-dir",
        default=None,
        help="Artefact directory with the trained model. "
        "Defaults to the most recent run under the config's artefact_dir.",
    )

    return parser


def _latest_run_dir(artefact_dir: str) -> str:
    """Return the most recent completed training run subdirectory."""
    if not os.path.isdir(artefact_dir):
        raise FileNotFoundError(
            f"No runs found: artefact directory '{artefact_dir}' does not exist. "
            "Run 'train' first."
        )
    candidates = [
        name
        for name in os.listdir(artefact_dir)
        if os.path.isdir(os.path.join(artefact_dir, name))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No runs found under '{artefact_dir}'. Run 'train' first."
        )
    # Prefer a completed training run (has model weights); ignore empty/inference
    # throwaway dirs that contain no artefacts. Timestamps sort lexicographically.
    complete = [
        name
        for name in candidates
        if os.path.isfile(os.path.join(artefact_dir, name, "model_weights.keras"))
    ]
    if complete:
        return os.path.join(artefact_dir, max(complete))
    return os.path.join(artefact_dir, max(candidates))


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        tracker = Experiment_Tracker(args.config)
        if args.command == "train":
            tracker.run_training()
            print("Training completed successfully.", file=sys.stdout)
        elif args.command == "infer":
            if args.run_dir is not None:
                run_dir = args.run_dir
            else:
                run_dir = _latest_run_dir(load_config(args.config).output.artefact_dir)
            predictions, mse_scores = tracker.run_inference(args.data, run_dir)
            print(f"Inference complete: {predictions.shape[0]} records classified.")
            return 0
        return 0
    except NetworkAnomalyBaseError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
