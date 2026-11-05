"""The values of config.json5 used by the code, and the constants of the game.

Everything here is a fixed value: what to run, folders, file names, columns, seed,
number of folds, and the stats and statuses of a Pokemon.
"""

import os
from pathlib import Path

from .config_loader import PROJECT_ROOT, SETTINGS

# Avoid a joblib warning on Windows when it cannot count physical cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


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
