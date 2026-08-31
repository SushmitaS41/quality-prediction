"""Train autoencoder(s) and save artifacts.

Usage:
    # Ensemble of 10 (default — stable latent vectors)
    python -m dissolution_pred.scripts.train_autoencoder

    # Single run
    python -m dissolution_pred.scripts.train_autoencoder --mode single

    # Custom ensemble size
    python -m dissolution_pred.scripts.train_autoencoder --n-ensemble 5
"""

import argparse
import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dissolution_pred.data_preparation.clean_and_split import prepare_data
from dissolution_pred.data_preparation.feature_engineering import (
    train_and_encode,
    train_ensemble,
)
from dissolution_pred.utils.plotting import plot_latent_target_correlation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train autoencoder(s) for latent feature extraction.",
    )
    default_data_dir = os.path.join(project_root, "..")
    parser.add_argument(
        "--data-dir", type=str, default=default_data_dir,
        help="Directory containing CSV/Excel data files (default: %(default)s)",
    )
    parser.add_argument(
        "--mode", type=str, default="ensemble", choices=["single", "ensemble"],
        help="single: 1 autoencoder. ensemble: N runs, averaged latent vectors (default: ensemble)",
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=10,
        help="Number of ensemble runs (default: 10)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.60,
        help="Fraction of batches for training (default: 0.60)",
    )
    parser.add_argument(
        "--target", type=str, default="FilteredArea1",
        help="Lab variable to predict — used for correlation plot (default: FilteredArea1)",
    )
    parser.add_argument(
        "--nan-threshold", type=float, default=0.90,
        help="Drop columns with NaN fraction above this (default: 0.90)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="trained_artifacts",
        help="Directory to save autoencoder artifacts (default: trained_artifacts)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    # ── Load & prepare data ──
    batch_df = pd.read_csv(os.path.join(data_dir, "batch_data.csv"))
    sensor_df = pd.read_csv(os.path.join(data_dir, "sensor_data.csv"))
    lab_df = pd.read_csv(os.path.join(data_dir, "lab_data.csv"))
    tagoffset_df = pd.read_excel(os.path.join(data_dir, "tag_offsets.xlsx"))

    data = prepare_data(
        batch_df, sensor_df, lab_df, tagoffset_df,
        frequency="1min",
        col_nan_threshold=args.nan_threshold,
        train_frac=args.train_frac,
        target_variable=args.target,
    )

    # ── Train autoencoder(s) ──
    if args.mode == "ensemble":
        print(f"\nTraining ensemble of {args.n_ensemble} autoencoders...")
        latent_train, _, _ = train_ensemble(
            data["train_df"],
            n_runs=args.n_ensemble,
        )
    else:
        print("\nTraining single autoencoder...")
        latent_train, _, _ = train_and_encode(data["train_df"])

    # ── Show correlation with target ──
    plot_latent_target_correlation(latent_train, data["target_df"])

    print(f"\nAutoencoder artifacts saved to {args.save_dir}/")
    print("Run train_pipeline.py with --ae-mode load to use these for model training.")


if __name__ == "__main__":
    main()
