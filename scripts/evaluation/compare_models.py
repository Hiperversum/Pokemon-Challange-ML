"""Compare all base models with stratified k-fold cross-validation.

Started by main.py with command: "compare" in config.json5 (reads run.models).
"""

from src.utility.config.config_constants import N_FOLDS
from src.utility.config.config_paths import get_output_dir
from src.utility.config.config_run import run_models
from src.evaluation.validation import compare_models
from src.models.catalog import MODEL_FACTORIES, build_model
from src.features.load_dataset import load_dataset
from src.utility.print.messages import print_saved, print_section, print_step, print_table


def main():
    """Build the features and print the cross-validation table."""
    names = run_models(MODEL_FACTORIES)
    X, y, _ = load_dataset()
    models = {name: build_model(name) for name in names}
    print_step(f"Cross-validating {len(models)} models with {N_FOLDS} folds")
    table = compare_models(models, X.values, y.values)
    out_path = get_output_dir() / "model_comparison.csv"
    table.to_csv(out_path, index=False)
    print_section("Results")
    print_table(table)
    print_saved(out_path)


if __name__ == "__main__":
    main()
