#!/usr/bin/env python
"""
Model quality gate.

Compares the new model MAE with the Production model MAE.
The CI check fails if MAE regresses by more than 5%.
"""

import json
import sys
from pathlib import Path


MAX_REGRESSION = 0.05


def load_new_mae(metrics_path: Path) -> float:
    """Load MAE from the new model evaluation metrics."""
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    return float(metrics["mae"])


def check_quality(new_mae: float, production_mae: float) -> bool:
    """Return True if the new model passes the quality gate."""

    if production_mae <= 0:
        raise ValueError("Production MAE must be greater than zero.")

    regression = (new_mae - production_mae) / production_mae

    print(f"Production MAE: {production_mae:.4f}")
    print(f"New model MAE:  {new_mae:.4f}")
    print(f"MAE regression: {regression * 100:.2f}%")
    print(f"Allowed regression: {MAX_REGRESSION * 100:.2f}%")

    if regression > MAX_REGRESSION:
        print("❌ Model quality gate FAILED.")
        print("The new model's MAE regressed by more than 5%.")
        return False

    print("✅ Model quality gate PASSED.")
    return True


def main() -> None:
    """Run the model quality gate."""

    metrics_path = Path("metrics/eval_metrics.json")

    if not metrics_path.exists():
        print(f"❌ Metrics file not found: {metrics_path}")
        sys.exit(1)

    if len(sys.argv) != 2:
        print("Usage: python scripts/model_quality_gate.py <production_mae>")
        sys.exit(1)

    production_mae = float(sys.argv[1])
    new_mae = load_new_mae(metrics_path)

    passed = check_quality(
        new_mae=new_mae,
        production_mae=production_mae,
    )

    if not passed:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
