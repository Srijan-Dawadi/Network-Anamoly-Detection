"""
main.py
=======
Command-line entry point for the network-anomaly autoencoder.

Usage:
    python main.py train --config configs/nsl_kdd_default.yaml
    python main.py infer  --config configs/nsl_kdd_default.yaml --data data/new_traffic.csv
"""

import argparse
import sys

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

    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        tracker = Experiment_Tracker(args.config)
        if args.command == "train":
            tracker.run_training()
            print("Training completed successfully.", file=sys.stdout)
        elif args.command == "infer":
            predictions, mse_scores = tracker.run_inference(args.data)
            print(f"Inference complete: {predictions.shape[0]} records classified.")
            return 0
        return 0
    except NetworkAnomalyBaseError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
