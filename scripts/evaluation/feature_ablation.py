"""Measure how much each feature group contributes: without each group, and with only that group.

Usage: python scripts/evaluation/feature_ablation.py [--model logreg]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.validation import cross_validate  # noqa: E402
from src.features.core.feature_builder import FeatureBuilder  # noqa: E402
from src.models.catalog import MODEL_FACTORIES  # noqa: E402
from src.models.presets import build_tuned_model  # noqa: E402
from src.features.load_dataset import load_dataset  # noqa: E402
from src.utility.print.messages import print_saved, print_scores, print_section, print_step, print_table  # noqa: E402


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logreg", choices=sorted(MODEL_FACTORIES))
    return parser.parse_args()


def main():
    """Cross-validate the tuned model with all groups, without each group, and with only one group."""
    args = parse_args()
    all_groups = FeatureBuilder.GROUPS
    settings = {"all": all_groups}
    # Remove one group at a time: how much does the model lose without it?
    for group in all_groups:
        settings[f"without_{group}"] = tuple(g for g in all_groups if g != group)
    # Keep one group only: how much can it do alone?
    for group in all_groups:
        settings[f"only_{group}"] = (group,)

    rows = []
    for position, (label, groups) in enumerate(settings.items(), start=1):
        print_step(f"[{position}/{len(settings)}] Setting: {label}")
        X, y, _ = load_dataset(groups=groups)
        result = cross_validate(build_tuned_model(args.model), X.values, y.values)
        rows.append({"setting": label, "n_features": X.shape[1],
                     "val_acc": result["val_acc"], "val_std": result["val_std"]})
        print_scores(f"{label} ({X.shape[1]} features)", result["train_acc"], result["val_acc"],
                     result["val_std"], width=40)

    table = pd.DataFrame(rows)
    out_path = config.get_output_dir() / f"ablation_{args.model}.csv"
    table.to_csv(out_path, index=False)
    print_section("Results")
    print_table(table)
    print_saved(out_path)


if __name__ == "__main__":
    main()
