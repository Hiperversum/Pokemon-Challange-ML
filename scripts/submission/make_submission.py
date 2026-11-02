"""Train a tuned model or ensemble on the full training set and write a submission file.

Usage:
    python scripts/submission/make_submission.py --model hist_gradient_boosting
    python scripts/submission/make_submission.py --model voting
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.catalog import MODEL_FACTORIES  # noqa: E402
from src.models.presets import ENSEMBLES, build_ensemble, build_tuned_model  # noqa: E402
from src.features.load_dataset import load_dataset  # noqa: E402
from src.models.submission import fit_and_predict, make_submission, matches_sample_submission  # noqa: E402
from src.utility.print.messages import print_info, print_saved, print_step, timed_step  # noqa: E402


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="hist_gradient_boosting",
                        choices=sorted(MODEL_FACTORIES) + sorted(ENSEMBLES))
    return parser.parse_args()


def main():
    """Fit the chosen model and save its test predictions."""
    args = parse_args()
    X_train, y_train, X_test = load_dataset()
    model = build_ensemble(args.model) if args.model in ENSEMBLES else build_tuned_model(args.model)
    print_step(f"Model: {args.model}")
    with timed_step("Training on the full train set and predicting the test battles"):
        predictions = fit_and_predict(model, X_train, y_train, X_test)
    path = make_submission(X_test.index, predictions, f"submission_{args.model}.csv")
    print_info("predictions", len(predictions))
    print_info("share of Player 1 wins", f"{predictions.mean():.3f}")
    print_info("same ids as sample", matches_sample_submission(X_test.index))
    print_saved(path)


if __name__ == "__main__":
    main()
