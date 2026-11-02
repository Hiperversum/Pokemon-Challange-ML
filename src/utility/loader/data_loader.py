"""Read the competition files from disk."""

import json

import pandas as pd # type: ignore

from ... import config


####################################################################################
# BATTLES (JSONL FILES)
####################################################################################

def load_jsonl(path):
    """Read a .jsonl file and return a list of dictionaries (one per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_battles(split, data_dir=None):
    """Load the battles of the given split ("train" or "test")."""
    data_dir = data_dir or config.get_data_dir()
    file_name = {"train": config.TRAIN_FILE, "test": config.TEST_FILE}[split]
    return load_jsonl(data_dir / file_name)


####################################################################################
# TABLES (CSV FILES)
####################################################################################

def load_type_chart(data_dir=None):
    """Load the Gen 2 type chart as a DataFrame (attacking_type, defending_type, multiplier)."""
    data_dir = data_dir or config.get_data_dir()
    return pd.read_csv(data_dir / config.TYPE_CHART_FILE)


def load_sample_submission(data_dir=None):
    """Load the sample submission file provided by the competition."""
    data_dir = data_dir or config.get_data_dir()
    return pd.read_csv(data_dir / config.SAMPLE_SUBMISSION_FILE)
