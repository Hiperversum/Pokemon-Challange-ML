"""Compare the tuned single models and the ensembles with stratified k-fold CV.

Usage: python scripts/evaluation/compare_tuned.py [--models logreg voting ...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.validation import compare_models  # noqa: E402
from src.models.presets import (  # noqa: E402
    ENSEMBLES, PCA_MODELS, TUNED_MODELS, build_named_model,
)
from src.features.load_dataset import load_dataset  # noqa: E402
from src.utility.print.messages import print_saved, print_section, print_step, print_table  # noqa: E402


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(TUNED_MODELS) + list(PCA_MODELS) + list(ENSEMBLES))
    return parser.parse_args()


def main():
    """Build the features and print the cross-validation table of the tuned models."""
    args = parse_args()
    X, y, _ = load_dataset()
    models = {name: build_named_model(name, X.columns) for name in args.models}
    print_step(f"Cross-validating {len(models)} tuned models and ensembles with {config.N_FOLDS} folds")
    table = compare_models(models, X.values, y.values)
    out_path = config.get_output_dir() / "tuned_comparison.csv"
    table.to_csv(out_path, index=False)
    print_section("Results")
    print_table(table)
    print_saved(out_path)


if __name__ == "__main__":
    main()
