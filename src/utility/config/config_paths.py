"""Find the folders of the project: where the data is read and where the outputs are written."""

from pathlib import Path

from .config_constants import (
    DATA_DIR_SETTING, KAGGLE_INPUT_DIR, KAGGLE_WORKING_DIR, LOCAL_DATA_DIR, OUTPUT_DIR,
    OUTPUT_DIR_SETTING, TRAIN_FILE,
)
from .config_loader import PROJECT_ROOT


####################################################################################
# PATHS WRITTEN IN config.json5
####################################################################################

def resolve_path(value):
    """Turn a path from config.json5 into an absolute one (relative paths start from the project root)."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


####################################################################################
# DATA FOLDER
####################################################################################

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


####################################################################################
# OUTPUT FOLDERS
####################################################################################

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


def get_output_dir(script):
    """Return the folder where one script writes its files: outputs/<script>_results.

    For example get_output_dir("ablation") is outputs/ablation_results.
    On Kaggle the files stay directly in the working folder, where the notebook output can be submitted.
    """
    root = get_output_root()
    out = root if root == KAGGLE_WORKING_DIR else root / f"{script}_results"
    out.mkdir(parents=True, exist_ok=True)
    return out
