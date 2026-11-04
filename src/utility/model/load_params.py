"""Read and write tuned_params.json: the best hyperparameters found by Optuna for each model.

The file lives in data/models/ and is saved in the repository (unlike the competition files
in data/), so the Kaggle notebook that clones the repository uses the same tuned models.
"""

import json

from ..config.config_loader import PROJECT_ROOT


####################################################################################
# FILE
####################################################################################

PARAMS_FILE = PROJECT_ROOT / "data" / "models" / "tuned_params.json"


####################################################################################
# READ
####################################################################################

def load_params():
    """Return the whole file: {model: {"params": {...}, "val_acc": ...}} (empty if the file does not exist)."""
    if not PARAMS_FILE.exists():
        return {}
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tuned_models():
    """Return {model: hyperparameters}, the form used by presets.py."""
    return {name: entry["params"] for name, entry in load_params().items()}


####################################################################################
# WRITE
####################################################################################

def save_params(name, params, val_acc):
    """Save the hyperparameters of one model, only if their accuracy beats the saved one.

    Return (saved, old_val_acc): old_val_acc is None when the model was not in the file yet.
    """
    entries = load_params()
    old_val_acc = entries.get(name, {}).get("val_acc")

    # A worse search (for example a quick test with few trials) never replaces a better one
    if old_val_acc is not None and val_acc <= old_val_acc:
        return False, old_val_acc

    # Write the new values; the other models stay as they are
    entries[name] = {"params": params, "val_acc": round(float(val_acc), 4)}
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4)
        f.write("\n")
    return True, old_val_acc
