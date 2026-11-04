"""Study our gradient descent logistic regression (Optimization slides).

1. Cost history table for several learning rates (batch GD) and for mini-batch GD.
2. Stratified k-fold accuracy of the GD model compared with sklearn's LogisticRegression.

Started by main.py with command: "gradient-descent" in config.json5.
"""

import pandas as pd

from src.utility.config.config_paths import get_output_dir
from src.evaluation.validation import cross_validate
from src.models.catalog import build_model
from src.features.load_dataset import load_dataset
from src.utility.print.messages import print_saved

LEARNING_RATES = [0.001, 0.01, 0.1, 1.0]
L2_VALUES = [0.0, 0.01, 0.1]
EPOCHS = 300


def cost_histories(X, y):
    """Fit the GD model with different settings and return the cost every 10 epochs, one column each."""
    table = pd.DataFrame({"epoch": range(0, EPOCHS + 1, 10)})

    # Batch GD: one step per epoch, keep the cost of every 10th epoch
    for lr in LEARNING_RATES:
        model = build_model("gd_logreg", learning_rate=lr, n_epochs=EPOCHS).fit(X, y)
        table[f"batch GD, alpha={lr}"] = model[-1].cost_history_[::10]

    # Mini-batch GD: many steps per epoch, so it runs 10 times fewer epochs (one per row)
    model = build_model("gd_logreg", learning_rate=0.01, n_epochs=EPOCHS // 10, batch_size=64).fit(X, y)
    table["mini-batch 64, alpha=0.01 (epoch = row / 10)"] = model[-1].cost_history_
    return table


def main():
    """Save the cost histories and compare the models with k-fold."""
    X, y, _ = load_dataset()
    X, y = X.values, y.values
    out_dir = get_output_dir()
    cost_path = out_dir / "gd_cost_history.csv"
    cost_histories(X, y).to_csv(cost_path, index=False)
    print_saved(cost_path)

    models = {"sklearn logreg C=0.01": build_model("logreg", C=0.01)}
    for l2 in L2_VALUES:
        models[f"gd_logreg l2={l2}"] = build_model("gd_logreg", learning_rate=0.1, n_epochs=EPOCHS, l2=l2)

    rows = []
    for name, model in models.items():
        result = cross_validate(model, X, y)
        rows.append({"model": name, "train_acc": result["train_acc"],
                     "val_acc": result["val_acc"], "val_std": result["val_std"]})
        print(f"{name:<24} train={result['train_acc']:.4f}  "
              f"val={result['val_acc']:.4f} +/- {result['val_std']:.4f}", flush=True)
    out_path = out_dir / "gradient_descent_experiment.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print_saved(out_path)


if __name__ == "__main__":
    main()
