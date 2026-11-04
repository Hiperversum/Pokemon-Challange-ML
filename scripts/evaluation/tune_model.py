"""Grid search the hyperparameters of one model with k-fold CV.

Started by main.py with command: "tune" in config.json5 (reads run.model).
"""

from src.utility.config.config_paths import get_output_dir
from src.utility.config.config_run import run_model
from src.evaluation.tuning import PARAM_GRIDS, grid_search
from src.features.load_dataset import load_dataset
from src.utility.print.messages import print_info, print_saved, print_section, print_step, print_table


def main():
    """Run the grid search and save the full results table."""
    model = run_model(sorted(PARAM_GRIDS))
    X, y, _ = load_dataset()
    print_step(f"Tuning {model}")
    best_params, table = grid_search(model, X.values, y.values)
    out_path = get_output_dir() / f"grid_{model}.csv"
    table.to_csv(out_path, index=False)
    print_section("Top 10 combinations")
    print_table(table.head(10))
    print_info("best parameters", best_params)
    print_saved(out_path)


if __name__ == "__main__":
    main()
