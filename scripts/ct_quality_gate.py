#!/usr/bin/env python
"""
Continuous Training quality gate.

Compares the candidate model MAE with the Production MAE.

The candidate is promoted only if it improves Production MAE
by at least the configured margin.

A rejected candidate is NOT considered a pipeline failure.
"""

import argparse
import json
import os
from pathlib import Path


DEFAULT_MARGIN = 0.05


def load_candidate_mae(metrics_path: Path) -> float:
    """Load candidate MAE from DVC evaluation metrics."""
    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    return float(metrics["mae"])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--production-mae",
        type=float,
        required=True,
        help="MAE of the current Production model.",
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("metrics/eval_metrics.json"),
        help="Path to candidate evaluation metrics.",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=float(
            os.getenv(
                "CT_MAE_MARGIN",
                DEFAULT_MARGIN,
            )
        ),
        help="Required improvement margin.",
    )

    args = parser.parse_args()

    if args.production_mae <= 0:
        raise ValueError("Production MAE must be greater than zero.")

    if args.margin < 0 or args.margin >= 1:
        raise ValueError("Margin must be between 0 and 1.")

    if not args.metrics.exists():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    candidate_mae = load_candidate_mae(args.metrics)

    required_mae = args.production_mae * (1 - args.margin)

    print("=" * 60)
    print("Continuous Training Quality Gate")
    print("=" * 60)
    print(f"Production MAE : {args.production_mae:.4f}")
    print(f"Candidate MAE  : {candidate_mae:.4f}")
    print(f"Required margin: {args.margin * 100:.2f}%")
    print(f"Maximum allowed candidate MAE: {required_mae:.4f}")
    print()

    if candidate_mae <= required_mae:
        print("✅ Candidate PASSED the promotion gate.")
        print("Candidate is better than Production by the required margin.")

        with open(
            os.environ.get("GITHUB_OUTPUT", "/dev/null"),
            "a",
            encoding="utf-8",
        ) as output:
            output.write("promote=true\n")

        return

    print("⚠️ Candidate REJECTED.")
    print("Candidate did not improve Production MAE " "by the required margin.")
    print("This is a normal model rejection, not a pipeline failure.")

    with open(
        os.environ.get("GITHUB_OUTPUT", "/dev/null"),
        "a",
        encoding="utf-8",
    ) as output:
        output.write("promote=false\n")


if __name__ == "__main__":
    main()
