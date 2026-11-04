"""Search the hyperparameters of one model with Optuna and k-fold CV.

Started by main.py with command: "optuna" in config.json5 (reads run.model and run.trials).
"""

from src.utility.config.config_paths import get_output_dir
from src.utility.config.config_run import run_model, run_trials
from src.evaluation.tuning import OPTUNA_SPACES, optuna_search
from src.features.load_dataset import load_dataset
from src.utility.print.messages import print_info, print_saved, print_section, print_step, print_table


def main():
    """Run the Optuna search and save the table of all the trials."""
    model = run_model(sorted(OPTUNA_SPACES))
    trials = run_trials()
    X, y, _ = load_dataset()
    print_step(f"Tuning {model} with Optuna")
    best_params, table = optuna_search(model, X.values, y.values, n_trials=trials)
    out_path = get_output_dir("optuna") / f"optuna_{model}.csv"
    table.to_csv(out_path, index=False)
    columns = ["number", "value"] + [c for c in table.columns if c.startswith("params_")]
    print_section("Top 10 trials")
    print_table(table[columns].head(10))
    print_info("best parameters", best_params)
    print_saved(out_path)


if __name__ == "__main__":
    main()
