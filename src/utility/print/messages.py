"""All the messages shown on the terminal: progress, model scores, tables and files.

(Only plain ASCII characters are used, so they work in any terminal.)
"""

import time
from contextlib import contextmanager

from ..config.config_constants import N_FOLDS, RANDOM_STATE
from ..config.config_paths import get_data_dir, get_output_root

LINE_WIDTH = 72


####################################################################################
# TIME
####################################################################################

def format_seconds(seconds):
    """Write a duration in a readable way (12s, 3m 05s, 1h 02m)."""
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


####################################################################################
# TITLES AND SECTIONS
####################################################################################

def print_title(text):
    """Show a big title between two lines of '='."""
    print("=" * LINE_WIDTH)
    print(text)
    print("=" * LINE_WIDTH, flush=True)


def print_section(text):
    """Show the title of a section of the output."""
    print(f"\n--- {text} ".ljust(LINE_WIDTH, "-"), flush=True)


####################################################################################
# PROGRESS MESSAGES
####################################################################################

def print_step(text):
    """Announce what the program is about to do."""
    print(f"> {text}", flush=True)


def print_info(label, value):
    """Show a setting or a number, aligned in two columns."""
    print(f"    {label:<22} {value}", flush=True)


@contextmanager
def timed_step(text):
    """Announce a step, run the code of the 'with' block and print how long it took."""
    print(f"> {text} ...", flush=True)
    start = time.time()
    yield
    print(f"    done in {format_seconds(time.time() - start)}", flush=True)


####################################################################################
# COMMANDS
####################################################################################

def print_run_header(command, script):
    """Show which command is going to run and with which settings."""
    print_title(f"python main.py: {command}")
    print_info("script", script)
    print_info("data folder", get_data_dir())
    print_info("output folder", get_output_root())
    print_info("seed / folds", f"{RANDOM_STATE} / {N_FOLDS}")


def print_commands(commands, chosen):
    """List the available commands and say which one is chosen in config.json5."""
    print("Available commands:")
    for name, script in commands.items():
        print(f"  {name:<18} {script}")
    print()
    print(f"Chosen in config.json5: {chosen}", flush=True)


def print_wrong_value(value, allowed):
    """Say that a model written in config.json5 cannot be used by this command, and list the valid ones."""
    print(f'"{value}" cannot be used by this command.')
    print(f"Valid names: {', '.join(allowed)}", flush=True)


####################################################################################
# MODEL EVALUATION
####################################################################################

def print_model_start(position, total, name):
    """Start the line of a model; the fold scores are added on the same line."""
    print(f"[{position}/{total}] {name:<24} folds: ", end="", flush=True)


def print_fold_done(score):
    """Add the validation accuracy of one fold to the current line."""
    print(f"{score:.3f} ", end="", flush=True)


def print_model_result(result, seconds):
    """Close the line of a model with its average scores and the time it took."""
    print(f"-> train={result['train_acc']:.4f}  val={result['val_acc']:.4f} "
          f"+/- {result['val_std']:.4f}  ({format_seconds(seconds)})", flush=True)


def print_trial(number, total, score, best):
    """Show the k-fold accuracy of one Optuna trial and the best one so far."""
    marker = "  <- new best" if score >= best else ""
    print(f"  trial {number:>3}/{total}  val={score:.4f}  best={best:.4f}{marker}", flush=True)


def print_params_update(name, new_val_acc, old_val_acc, saved):
    """Say whether tuned_params.json now holds the new hyperparameters of a model."""
    if not saved:
        print_step(f"tuned_params.json: {name} kept (new {new_val_acc:.4f} <= saved {old_val_acc:.4f})")
    elif old_val_acc is None:
        print_step(f"tuned_params.json: {name} added ({new_val_acc:.4f})")
    else:
        print_step(f"tuned_params.json: {name} updated ({old_val_acc:.4f} -> {new_val_acc:.4f})")


def print_scores(label, train_acc, val_acc, val_std, width=26):
    """Show the scores of one experiment on a single line."""
    print(f"  {label:<{width}} train={train_acc:.4f}  val={val_acc:.4f} +/- {val_std:.4f}", flush=True)


####################################################################################
# TABLES AND FILES
####################################################################################

def print_table(table):
    """Show a pandas table without the row numbers."""
    print(table.to_string(index=False), flush=True)


def print_saved(path):
    """Tell where a file has been written."""
    print(f"> Saved: {path}", flush=True)


def print_total_time(seconds):
    """Close the run with the total time."""
    print(f"\nFinished in {format_seconds(seconds)}.", flush=True)
