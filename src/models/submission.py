"""Train a model on the full training set and write the file to submit on Kaggle."""

import pandas as pd

from ..utility.config.config_constants import ID_COLUMN, TARGET
from ..utility.config.config_paths import get_output_dir
from ..utility.data.data_loader import load_sample_submission


####################################################################################
# TRAINING AND PREDICTION
####################################################################################

def fit_and_predict(model, X_train, y_train, X_test):
    """Fit a model on the full training set and predict the test battles."""
    model.fit(X_train.values, y_train.values)
    return model.predict(X_test.values).astype(int)


####################################################################################
# SUBMISSION FILE
####################################################################################

def make_submission(test_ids, predictions, file_name="submission.csv"):
    """Write predictions in the sample_submission format and return the file path."""
    # One row per test battle: its id and the predicted winner (1 = Player 1)
    submission = pd.DataFrame({ID_COLUMN: test_ids, TARGET: predictions})
    path = get_output_dir("submissions") / file_name
    submission.to_csv(path, index=False)
    return path


def matches_sample_submission(test_ids):
    """Return True if the submission has exactly the battle_ids of sample_submission.csv."""
    sample_ids = load_sample_submission()[ID_COLUMN]
    return len(test_ids) == len(sample_ids) and set(test_ids) == set(sample_ids)
