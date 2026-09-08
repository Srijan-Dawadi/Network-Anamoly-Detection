"""
main.py
=======
Command-line entry point for the network-anomaly autoencoder.

Usage:
    python main.py train --config configs/nsl_kdd_default.yaml
    python main.py infer  --config configs/nsl_kdd_default.yaml --data data/new_traffic.csv
"""

import argparse
import json
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

    report_parser = subparsers.add_parser(
        "report", help="Print a summary of a completed training run."
    )
    report_parser.add_argument(
        "--config", required=True, help="Path to the YAML experiment configuration."
    )
    report_parser.add_argument(
        "--run-dir",
        default=None,
        help="Artefact directory to summarise. "
        "Defaults to the most recent completed run under the config's artefact_dir.",
    )
    report_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the run's PNG plots with the system default viewer.",
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


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _print_experiment_report(run_dir: str, open_plots: bool = False) -> None:
    """Print a human-readable summary of a completed training run."""
    metrics_path = os.path.join(run_dir, "metrics.json")
    threshold_path = os.path.join(run_dir, "threshold.json")

    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(
            f"No report available for run '{run_dir}': '{metrics_path}' is missing. "
            "Run 'train' to produce a full experiment first."
        )
    metrics = _load_json(metrics_path)
    threshold = _load_json(threshold_path) if os.path.isfile(threshold_path) else {}

    precision_n = metrics.get("precision_normal")
    recall_n = metrics.get("recall_normal")
    f1_n = metrics.get("f1_normal")
    precision_a = metrics.get("precision_anomalous")
    recall_a = metrics.get("recall_anomalous")
    f1_a = metrics.get("f1_anomalous")
    accuracy = metrics.get("accuracy")
    auc = metrics.get("auc_roc")
    train_secs = metrics.get("training_time_secs")

    acc_s = f"{100 * accuracy:.2f}%" if accuracy is not None else "N/A"
    auc_s = f"{auc:.4f}" if auc is not None else "N/A"
    thr_s = f"{threshold['threshold']:.6f}" if "threshold" in threshold else "N/A"
    pct_s = f"{threshold.get('percentile', 'N/A')}th pct"
    mean_s = f"{threshold['mse_mean']:.6f}" if "mse_mean" in threshold else "N/A"
    std_s = f"{threshold['mse_std']:.6f}" if "mse_std" in threshold else "N/A"
    time_s = f"{train_secs:.2f}s" if isinstance(train_secs, (int, float)) else "N/A"

    def _pct(v):
        return f"{100 * v:.2f}%" if v is not None else "N/A"

    bar = "=" * 60
    print(bar)
    print(f"EXPERIMENT REPORT - {run_dir}")
    print(bar)
    print(f"  Accuracy           {acc_s}")
    print(f"  AUC-ROC            {auc_s}")
    print(f"  Threshold          {thr_s}  ({pct_s})")
    print(f"  MSE mean / std     {mean_s} / {std_s}")
    print(f"  Training time      {time_s}")
    print("-" * 60)
    print(f"  {'':<10}{'Normal':>12}{'Anomalous':>12}")
    print(f"  {'Precision':<10}{_pct(precision_n):>12}{_pct(precision_a):>12}")
    print(f"  {'Recall':<10}{_pct(recall_n):>12}{_pct(recall_a):>12}")
    print(f"  {'F1':<10}{_pct(f1_n):>12}{_pct(f1_a):>12}")
    print(bar)

    plots = [
        name
        for name in ("confusion_matrix.png", "roc_curve.png", "re_histogram.png")
        if os.path.isfile(os.path.join(run_dir, name))
    ]
    if plots:
        print("Plots:")
        for name in plots:
            print(f"  - {os.path.join(run_dir, name)}")
        if not open_plots:
            print("(Use --open to view them.)")

    if open_plots and hasattr(os, "startfile"):
        for name in plots:
            os.startfile(os.path.join(run_dir, name))  # type: ignore[attr-defined]


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
        elif args.command == "report":
            if args.run_dir is not None:
                run_dir = args.run_dir
            else:
                run_dir = _latest_run_dir(load_config(args.config).output.artefact_dir)
            _print_experiment_report(run_dir, open_plots=args.open)
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
