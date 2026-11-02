"""Test polynomial features for logistic regression (Linear Regression slides, feature maps).

Squares and pairwise products of the Player 1 minus Player 2 differences are added,
so a linear model can learn curved decision boundaries.

Usage: python scripts/experiments/polynomial_experiment.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.validation import cross_validate  # noqa: E402
from src.models.catalog import build_model  # noqa: E402
from src.models.catalog import polynomial_logistic_regression  # noqa: E402
from src.features.load_dataset import load_dataset  # noqa: E402
from src.utility.print.messages import print_saved  # noqa: E402

CORE_DIFFS = ["hp_remaining_diff", "fainted_diff", "strength_left_diff",
              "hp_lost_diff", "statused_diff", "alive_hp_mean_diff"]
C_VALUES = [0.003, 0.01, 0.03]


def main():
    """Compare logistic regression with and without polynomial features."""
    X, y, _ = load_dataset()
    core = [X.columns.get_loc(c) for c in CORE_DIFFS]
    all_diffs = [i for i, c in enumerate(X.columns) if c.endswith("_diff") and not c.startswith("bag_")]
    print(f"Polynomial base: {len(core)} core differences, {len(all_diffs)} differences in total")

    models = {}
    for C in C_VALUES:
        models[f"logreg C={C}"] = build_model("logreg", C=C)
        models[f"poly2 core C={C}"] = polynomial_logistic_regression(core, degree=2, C=C)
        models[f"poly2 all diffs C={C}"] = polynomial_logistic_regression(all_diffs, degree=2, C=C)

    rows = []
    for name, model in models.items():
        result = cross_validate(model, X.values, y.values)
        rows.append({"model": name, "train_acc": result["train_acc"],
                     "val_acc": result["val_acc"], "val_std": result["val_std"]})
        print(f"{name:<26} train={result['train_acc']:.4f}  "
              f"val={result['val_acc']:.4f} +/- {result['val_std']:.4f}", flush=True)
    out_path = config.get_output_dir() / "polynomial_experiment.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print_saved(out_path)


if __name__ == "__main__":
    main()
