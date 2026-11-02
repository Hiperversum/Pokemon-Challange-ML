"""Search the hyperparameters of one model with Optuna and k-fold CV.

Usage: python scripts/evaluation/optuna_model.py --model catboost [--trials 50]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.tuning import OPTUNA_SPACES, optuna_search  # noqa: E402
from src.features.load_dataset import load_dataset  # noqa: E402
from src.utility.print.messages import print_info, print_saved, print_section, print_step, print_table  # noqa: E402


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(OPTUNA_SPACES))
    parser.add_argument("--trials", type=int, default=50)
    return parser.parse_args()


def main():
    """Run the Optuna search and save the table of all the trials."""
    args = parse_args()
    X, y, _ = load_dataset()
    print_step(f"Tuning {args.model} with Optuna")
    best_params, table = optuna_search(args.model, X.values, y.values, n_trials=args.trials)
    out_path = config.get_output_dir() / f"optuna_{args.model}.csv"
    table.to_csv(out_path, index=False)
    columns = ["number", "value"] + [c for c in table.columns if c.startswith("params_")]
    print_section("Top 10 trials")
    print_table(table[columns].head(10))
    print_info("best parameters", best_params)
    print_saved(out_path)


if __name__ == "__main__":
    main()
