"""Single entry point of the project: python main.py

What to run is chosen in config.json5 (section "run"), not on the command line.
"""

from src.utility.config.config_run import choose_command, run_script


def main():
    """Run the command chosen in config.json5."""
    # The command written in config.json5 and the script that runs it
    command, script = choose_command()

    # Run the script: it reads its model, models or trials from config.json5 by itself
    run_script(command, script)


if __name__ == "__main__":
    main()
