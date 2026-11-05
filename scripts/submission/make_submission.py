"""Train a tuned model or ensemble on the full training set and write a submission file.

Started by main.py with command: "submit" in config.json5 (reads run.model).
"""

from src.utility.config.config_run import run_model
from src.models.core.model_builder import ENSEMBLES, MODEL_FACTORIES, build_ensemble, build_tuned_model
from src.features.load_dataset import load_dataset
from src.models.submission import fit_and_predict, make_submission, matches_sample_submission
from src.utility.print.messages import print_info, print_saved, print_step, timed_step


####################################################################################
# SUBMISSION FOR KAGGLE
####################################################################################

def main():
    """Fit the chosen model and save its test predictions."""

    #---------------------------
    # MODEL AND FEATURES
    #---------------------------

    # The model written in config.json5: a single model or an ensemble
    name = run_model(sorted(MODEL_FACTORIES) + sorted(ENSEMBLES))
    # Here the test features are needed too
    X_train, y_train, X_test = load_dataset()
    model = build_ensemble(name) if name in ENSEMBLES else build_tuned_model(name)
    print_step(f"Model: {name}")

    #---------------------------
    # TRAIN ON THE WHOLE TRAIN SET AND PREDICT THE TEST BATTLES
    #---------------------------

    with timed_step("Training on the full train set and predicting the test battles"):
        predictions = fit_and_predict(model, X_train, y_train, X_test)

    #---------------------------
    # WRITE THE FILE AND CHECK IT
    #---------------------------

    path = make_submission(X_test.index, predictions, f"submission_{name}.csv")
    print_info("predictions", len(predictions))
    # Sanity check: the share of test battles predicted as Player 1 wins
    print_info("share of Player 1 wins", f"{predictions.mean():.3f}")
    print_info("same ids as sample", matches_sample_submission(X_test.index))
    print_saved(path)


if __name__ == "__main__":
    main()
