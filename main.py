"""Single entry point of the project: python main.py

What to run is chosen in config.json5 (section "run"), not on the command line.
"""

from src import config
from src.utility.command import choose_command, options_to_arguments, run_script


def main():
    """Run the command chosen in config.json5."""
    # The command, its script and the values of "run" it reads, as written in config.json5
    command, script, names = choose_command()

    # Those values, written as command line arguments
    arguments = options_to_arguments(names, config.RUN_SETTINGS)

    # Run the script
    run_script(command, script, arguments)


if __name__ == "__main__":
    main()
