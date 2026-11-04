"""Search the hyperparameters of one model with Optuna and k-fold CV.

The best hyperparameters are saved in data/models/tuned_params.json when they beat the saved ones.

Started by main.py with command: "tune" in config.json5 (reads run.model and run.trials).
"""

from src.utility.config.config_paths import get_output_dir
from src.utility.config.config_run import run_model, run_trials
from src.utility.model.load_params import save_params
from src.evaluation.tuning import OPTUNA_SPACES, optuna_search
from src.features.load_dataset import load_dataset
from src.utility.print.messages import (
    print_info, print_params_update, print_saved, print_section, print_step, print_table,
)


####################################################################################
# TUNE ONE MODEL WITH OPTUNA
####################################################################################

def main():
    """Run the Optuna search, save the table of all the trials and the best hyperparameters."""

    #---------------------------
    # MODEL, TRIALS AND FEATURES
    #---------------------------

    # The model and the number of trials written in config.json5
    model = run_model(sorted(OPTUNA_SPACES))
    trials = run_trials()
    X, y, _ = load_dataset()

    #---------------------------
    # SEARCH: each trial is one combination scored with k-fold CV
    #---------------------------

    print_step(f"Tuning {model} with Optuna")
    best_params, table = optuna_search(model, X.values, y.values, n_trials=trials)

    #---------------------------
    # SAVE AND SHOW THE TRIALS
    #---------------------------

    out_path = get_output_dir("tune") / f"tune_{model}.csv"
    table.to_csv(out_path, index=False)
    # Only the trial number, its accuracy and the values it tried
    columns = ["number", "value"] + [c for c in table.columns if c.startswith("params_")]
    print_section("Top 10 trials")
    print_table(table[columns].head(10))
    print_info("best parameters", best_params)
    print_saved(out_path)

    #---------------------------
    # UPDATE tuned_params.json (only if the new accuracy is higher)
    #---------------------------

    # The table is sorted from the best trial, so the first value is the best accuracy
    best_val_acc = table["value"].iloc[0]
    saved, old_val_acc = save_params(model, best_params, best_val_acc)
    print_params_update(model, best_val_acc, old_val_acc, saved)


if __name__ == "__main__":
    main()
