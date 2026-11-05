"""Start the command written in config.json5, and give the scripts the values of the "run" section."""

import runpy
import sys
import time

from ..print.messages import print_commands, print_run_header, print_total_time, print_wrong_value
from .config_constants import COMMANDS, RUN_COMMAND, RUN_SETTINGS
from .config_paths import resolve_path


####################################################################################
# RUN THE COMMAND (used by main.py)
####################################################################################

def choose_command():
    """Return (command, script) of the command written in config.json5."""
    # Check that the command is known, and show the available commands if not
    if RUN_COMMAND not in COMMANDS:
        print_commands(COMMANDS, RUN_COMMAND)
        sys.exit(1)

    # The script that runs the command
    return RUN_COMMAND, COMMANDS[RUN_COMMAND]


def run_script(command, script):
    """Show the header, run the script and print how long it took."""
    print_run_header(command, script)

    # Run the script in its own __main__ module, so that its main() starts
    start = time.time()
    runpy.run_path(str(resolve_path(script)), run_name="__main__")

    # Print how long the command took to run
    print_total_time(time.time() - start)


####################################################################################
# VALUES OF THE RUN SECTION (used by the scripts)
####################################################################################

def run_model(allowed):
    """Return run.model of config.json5, after checking that the command can use it."""
    model = RUN_SETTINGS["model"]

    # Stop with the list of valid names if this command cannot use it
    if model not in allowed:
        print_wrong_value(model, allowed)
        sys.exit(1)
    return model


def run_models(default, allowed=None):
    """Return run.models of config.json5 ([] = the default models of the command), after checking them."""
    models = RUN_SETTINGS["models"] or list(default)
    allowed = list(allowed or default)

    # Stop at the first name this command cannot use
    for model in models:
        if model not in allowed:
            print_wrong_value(model, allowed)
            sys.exit(1)
    return models


def run_trials():
    """Return run.trials of config.json5: how many combinations Optuna tries."""
    return RUN_SETTINGS["trials"]
