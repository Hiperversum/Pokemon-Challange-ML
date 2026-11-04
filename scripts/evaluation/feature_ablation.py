"""Measure how much each feature group contributes: without each group, and with only that group.

Started by main.py with command: "ablation" in config.json5 (reads run.model).
"""

import pandas as pd

from src.utility.config.config_paths import get_output_dir
from src.utility.config.config_run import run_model
from src.evaluation.validation import cross_validate
from src.features.core.feature_builder import FeatureBuilder
from src.models.catalog import MODEL_FACTORIES
from src.models.presets import build_tuned_model
from src.features.load_dataset import load_dataset
from src.utility.print.messages import print_saved, print_scores, print_section, print_step, print_table


def main():
    """Cross-validate the tuned model with all groups, without each group, and with only one group."""
    model = run_model(sorted(MODEL_FACTORIES))
    print_step(f"Ablation of the feature groups with {model}")
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
        result = cross_validate(build_tuned_model(model), X.values, y.values)
        rows.append({"model": model, "setting": label, "n_features": X.shape[1],
                     "val_acc": result["val_acc"], "val_std": result["val_std"]})
        print_scores(f"{label} ({X.shape[1]} features)", result["train_acc"], result["val_acc"],
                     result["val_std"], width=40)

    table = pd.DataFrame(rows)
    out_path = get_output_dir("ablation") / f"ablation_{model}.csv"
    table.to_csv(out_path, index=False)
    print_section("Results")
    print_table(table)
    print_saved(out_path)


if __name__ == "__main__":
    main()
