"""The three steps of main.py: choose the command, build its arguments, run its script."""

import runpy
import sys
import time

from .. import config
from .print.messages import print_commands, print_run_header, print_total_time


####################################################################################
# COMMAND CHOSEN IN config.json5
####################################################################################

def choose_command():
    """Return (command, script, names of the run values it reads) of the command in config.json5."""
    command = config.RUN_COMMAND

    # Check that the command is known, and show the available commands if not
    if command not in config.COMMANDS:
        print_commands(config.COMMANDS, command)
        sys.exit(1)

    # The script that runs the command and the values of "run" it needs
    entry = config.COMMANDS[command]
    return command, entry["script"], entry["reads"]


####################################################################################
# RUN VALUES -> COMMAND LINE ARGUMENTS
####################################################################################

def options_to_arguments(names, settings):
    """Turn the run values a command reads into command line arguments.

    model: "voting" -> ["--model", "voting"]
    models: ["logreg", "knn"] -> ["--models", "logreg", "knn"]
    models: [] -> nothing, so the script uses its default (all the models)
    """
    arguments = []
    for name in names:
        value = settings[name]

        # An empty list means "use the default of the script"
        if isinstance(value, list):
            if value:
                arguments += [f"--{name}"] + [str(v) for v in value]
        else:
            arguments += [f"--{name}", str(value)]
    return arguments


####################################################################################
# RUN THE SCRIPT
####################################################################################

def run_script(command, script, arguments):
    """Show the header, run the script as if it was started alone, and print how long it took."""
    print_run_header(command, script, arguments)

    # Run the script with the arguments, in its own __main__ module
    start = time.time()
    sys.argv = [script] + arguments
    runpy.run_path(str(config.resolve_path(script)), run_name="__main__")

    # Print how long the command took to run
    print_total_time(time.time() - start)
