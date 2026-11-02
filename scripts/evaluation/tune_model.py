"""Grid search the hyperparameters of one model with k-fold CV.

Usage: python scripts/evaluation/tune_model.py --model logreg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.tuning import PARAM_GRIDS, grid_search  # noqa: E402
from src.features.load_dataset import load_dataset  # noqa: E402
from src.utility.print.messages import print_info, print_saved, print_section, print_step, print_table  # noqa: E402


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(PARAM_GRIDS))
    return parser.parse_args()


def main():
    """Run the grid search and save the full results table."""
    args = parse_args()
    X, y, _ = load_dataset()
    print_step(f"Tuning {args.model}")
    best_params, table = grid_search(args.model, X.values, y.values)
    out_path = config.get_output_dir() / f"grid_{args.model}.csv"
    table.to_csv(out_path, index=False)
    print_section("Top 10 combinations")
    print_table(table.head(10))
    print_info("best parameters", best_params)
    print_saved(out_path)


if __name__ == "__main__":
    main()
