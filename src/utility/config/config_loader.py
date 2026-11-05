"""Read config.json5, the settings file of the project, into a dictionary."""

from pathlib import Path

import json5


####################################################################################
# SETTINGS FILE
####################################################################################

# The project folder (three levels above this file: config -> utility -> src -> project)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_FILE = PROJECT_ROOT / "config.json5"


####################################################################################
# READ THE FILE
####################################################################################

def load_config(path):
    """Read the settings file (config.json5) and return it as a dictionary.

    JSON5 allows comments (lines starting with //) inside the file.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return json5.load(f)


# The whole content of config.json5, read once when the project starts
SETTINGS = load_config(CONFIG_FILE)
