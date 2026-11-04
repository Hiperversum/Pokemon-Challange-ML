"""Train a tuned model or ensemble on the full training set and write a submission file.

Started by main.py with command: "submit" in config.json5 (reads run.model).
"""

from src.utility.config.config_run import run_model
from src.models.catalog import MODEL_FACTORIES
from src.models.presets import ENSEMBLES, build_ensemble, build_tuned_model
from src.features.load_dataset import load_dataset
from src.models.submission import fit_and_predict, make_submission, matches_sample_submission
from src.utility.print.messages import print_info, print_saved, print_step, timed_step


def main():
    """Fit the chosen model and save its test predictions."""
    name = run_model(sorted(MODEL_FACTORIES) + sorted(ENSEMBLES))
    X_train, y_train, X_test = load_dataset()
    model = build_ensemble(name) if name in ENSEMBLES else build_tuned_model(name)
    print_step(f"Model: {name}")
    with timed_step("Training on the full train set and predicting the test battles"):
        predictions = fit_and_predict(model, X_train, y_train, X_test)
    path = make_submission(X_test.index, predictions, f"submission_{name}.csv")
    print_info("predictions", len(predictions))
    print_info("share of Player 1 wins", f"{predictions.mean():.3f}")
    print_info("same ids as sample", matches_sample_submission(X_test.index))
    print_saved(path)


if __name__ == "__main__":
    main()
