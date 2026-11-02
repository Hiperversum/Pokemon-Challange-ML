"""Project-wide paths and constants.

The settings that change from run to run (data and output folders, file names,
seed, number of folds, command to run) are read from config.json5 in the project root.
"""

import os
from pathlib import Path

from .utility.loader.config_loader import load_config

# Avoid a joblib warning on Windows when it cannot count physical cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


####################################################################################
# SETTINGS FILE
####################################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config.json5"

# The whole content of config.json5, as a dictionary
SETTINGS = load_config(CONFIG_FILE)


####################################################################################
# WHAT TO RUN
####################################################################################

# The "run" section (command, model, models, trials) and the commands it can choose from
RUN_SETTINGS = SETTINGS["run"]
RUN_COMMAND = RUN_SETTINGS["command"]
COMMANDS = SETTINGS["commands"]


####################################################################################
# FOLDERS
####################################################################################

# null in config.json5 means "find the folder automatically"
DATA_DIR_SETTING = SETTINGS["dataset"]["data_dir"]
OUTPUT_DIR_SETTING = SETTINGS["output"]["dir"]

LOCAL_DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")


####################################################################################
# DATASET FILES AND COLUMNS
####################################################################################

TRAIN_FILE = SETTINGS["dataset"]["train"]
TEST_FILE = SETTINGS["dataset"]["test"]
TYPE_CHART_FILE = SETTINGS["dataset"]["type_chart"]
SAMPLE_SUBMISSION_FILE = SETTINGS["dataset"]["sample_submission"]

TARGET = SETTINGS["dataset"]["target_column"]
ID_COLUMN = SETTINGS["dataset"]["id_column"]


####################################################################################
# VALIDATION
####################################################################################

RANDOM_STATE = SETTINGS["validation"]["random_state"]
N_FOLDS = SETTINGS["validation"]["n_folds"]


####################################################################################
# CONSTANTS OF THE GAME
####################################################################################

STATS = ["hp", "atk", "def", "spa", "spd", "spe"]
STATUSES = ["slp", "par", "frz", "tox", "psn", "brn"]
TEAM_SIZE = 6


####################################################################################
# FUNCTIONS
####################################################################################

def resolve_path(value):
    """Turn a path from config.json5 into an absolute one (relative paths start from the project root)."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_data_dir():
    """Return the folder that contains the competition files.

    Order of lookup: data_dir in config.json5, Kaggle input folder,
    local data/ folder of the repository.
    """
    if DATA_DIR_SETTING:
        return resolve_path(DATA_DIR_SETTING)
    if KAGGLE_INPUT_DIR.exists():
        matches = sorted(KAGGLE_INPUT_DIR.rglob(TRAIN_FILE))
        if matches:
            return matches[0].parent
    return LOCAL_DATA_DIR


def get_output_root():
    """Return the main output folder.

    Order of lookup: output dir in config.json5, Kaggle working folder,
    local outputs/ folder of the repository.
    """
    if OUTPUT_DIR_SETTING:
        return resolve_path(OUTPUT_DIR_SETTING)
    if KAGGLE_WORKING_DIR.exists():
        return KAGGLE_WORKING_DIR
    return OUTPUT_DIR


def get_output_dir(kind="results"):
    """Return the folder of one kind of output: "results" (tables) or "submissions" (csv for Kaggle).

    On Kaggle the files stay directly in the working folder, where the notebook output can be submitted.
    """
    root = get_output_root()
    out = root if root == KAGGLE_WORKING_DIR else root / kind
    out.mkdir(parents=True, exist_ok=True)
    return out
