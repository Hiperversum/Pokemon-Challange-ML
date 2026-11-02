"""Read the project settings file."""

from pathlib import Path

import json5


def load_config(path):
    """Read the settings file (config.json5) and return it as a dictionary.

    JSON5 allows comments (lines starting with //) inside the file.
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        return json5.load(f)
