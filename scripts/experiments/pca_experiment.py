"""Measure the effect of PCA on the models with stratified k-fold cross-validation.

Two variants are compared with the models without PCA:
- PCA on all the features;
- PCA only on the bag-of-tokens columns (species and moves counts).

Usage: python scripts/experiments/pca_experiment.py [--models logreg knn ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402
from src.evaluation.validation import cross_validate  # noqa: E402
from src.models.catalog import build_model  # noqa: E402
from src.models.catalog import pca_on_columns, scaled  # noqa: E402
from src.features.load_dataset import load_dataset  # noqa: E402
from src.utility.print.messages import print_saved  # noqa: E402

ALL_COMPONENTS = [5, 10, 20, 50, 100, 200]
BAG_COMPONENTS = [5, 10, 20, 50]


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["logreg", "knn", "hist_gradient_boosting"])
    return parser.parse_args()


def explained_variance_summary(X):
    """Print how many components keep 80%, 90%, 95% and 99% of the variance."""
    pca = PCA(random_state=config.RANDOM_STATE).fit(StandardScaler().fit_transform(X))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for share in (0.80, 0.90, 0.95, 0.99):
        n = int(np.searchsorted(cumulative, share) + 1)
        print(f"{share:.0%} of the variance kept with {n} components (out of {X.shape[1]})")


def with_pca(name, n_components):
    """Build a model with a PCA step on all features (scaled models already support it)."""
    if name in ("logreg", "knn"):
        return build_model(name, n_components=n_components)
    return scaled(build_model(name), n_components)


def model_variants(name, bag_columns):
    """Yield (variant, n_components, model): no PCA, PCA on all features, PCA on bags only."""
    yield "no_pca", None, build_model(name)
    for n in ALL_COMPONENTS:
        yield "pca_all", n, with_pca(name, n)
    for n in BAG_COMPONENTS:
        yield "pca_bags", n, pca_on_columns(build_model(name), bag_columns, n)


def main():
    """Build the features, run every variant and save the comparison table."""
    args = parse_args()
    X, y, _ = load_dataset()
    bag_columns = [i for i, col in enumerate(X.columns) if col.startswith("bag_")]
    print(f"Features: {X.shape[1]} ({len(bag_columns)} bag columns)  Train battles: {X.shape[0]}")
    explained_variance_summary(X.values)

    rows = []
    for name in args.models:
        for variant, n, model in model_variants(name, bag_columns):
            result = cross_validate(model, X.values, y.values)
            rows.append({"model": name, "variant": variant, "n_components": n,
                         "train_acc": result["train_acc"], "val_acc": result["val_acc"],
                         "val_std": result["val_std"]})
            print(f"{name:<24} {variant:<9} n={str(n):<5} train={result['train_acc']:.4f}  "
                  f"val={result['val_acc']:.4f} +/- {result['val_std']:.4f}", flush=True)

    table = pd.DataFrame(rows)
    out_path = config.get_output_dir() / "pca_experiment.csv"
    table.to_csv(out_path, index=False)
    print_saved(out_path)


if __name__ == "__main__":
    main()
