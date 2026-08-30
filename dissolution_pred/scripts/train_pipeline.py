"""Model training script — loads saved AE artifacts, builds features, trains models.

Usage:
    # First train the autoencoder:
    python -m dissolution_pred.scripts.train_autoencoder

    # Then train models:
    python -m dissolution_pred.scripts.train_pipeline
"""

import argparse
import os
import sys
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dissolution_pred.data_preparation.clean_and_split import prepare_data
from dissolution_pred.data_preparation.feature_engineering import prepare_features
from dissolution_pred.models.model_training import train_models
from dissolution_pred.utils.plotting import (
    plot_latent_target_correlation,
    plot_feature_importance,
    plot_results_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train dissolution quality prediction models.",
    )
    default_data_dir = os.path.join(project_root, "..")
    parser.add_argument(
        "--data-dir", type=str, default=default_data_dir,
        help="Directory containing batch_data.csv, lab_data.csv, "
             "sensor_data.csv, tag_offsets.xlsx  (default: %(default)s)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.60,
        help="Fraction of batches for training (default: 0.60)",
    )
    parser.add_argument(
        "--target", type=str, default="FilteredArea1",
        help="Lab variable to predict (default: FilteredArea1)",
    )
    parser.add_argument(
        "--nan-threshold", type=float, default=0.90,
        help="Drop columns with NaN fraction above this (default: 0.90)",
    )
    parser.add_argument(
        "--ae-save-dir", type=str, default="trained_autoencoder",
        help="Directory with saved autoencoder artifacts (default: trained_autoencoder)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)

    # ── Data paths ──
    batch_data_path = os.path.join(data_dir, "batch_data.csv")
    lab_data_path = os.path.join(data_dir, "lab_data.csv")
    sensor_data_path = os.path.join(data_dir, "sensor_data.csv")
    tagoffset_path = os.path.join(data_dir, "tag_offsets.xlsx")

    for p in [batch_data_path, lab_data_path, sensor_data_path, tagoffset_path]:
        if not os.path.isfile(p):
            sys.exit(f"ERROR: File not found: {p}")

    # ── Load raw data ──
    print(f"Loading raw data from {data_dir}/")
    batch_df = pd.read_csv(batch_data_path)
    sensor_df = pd.read_csv(sensor_data_path)
    lab_df = pd.read_csv(lab_data_path)
    tagoffset_df = pd.read_excel(tagoffset_path)
    print(f"  batch:  {batch_df.shape}")
    print(f"  sensor: {sensor_df.shape}")
    print(f"  lab:    {lab_df.shape}")
    print(f"  offsets: {tagoffset_df.shape}")

    # ══════════════════════════════════════════════════════════
    # PART 1: Data Preparation
    # ══════════════════════════════════════════════════════════
    data = prepare_data(
        batch_df, sensor_df, lab_df, tagoffset_df,
        frequency="1min",
        col_nan_threshold=args.nan_threshold,
        train_frac=args.train_frac,
        target_variable=args.target,
    )

    # ══════════════════════════════════════════════════════════
    # PART 2: Latent Variable Preparation
    # ══════════════════════════════════════════════════════════
    print()
    features = prepare_features(
        train_df=data["train_df"],
        test_df=data["test_df"],
        target_df=data["target_df"],
        batch_lab_merged=data["batch_lab_merged"],
        ae_save_dir=args.ae_save_dir,
        ae_mode="single",
    )

    # Plot latent-target correlations
    plot_latent_target_correlation(
        features["latent_train"],
        data["target_df"],
    )

    # ══════════════════════════════════════════════════════════
    # PART 3: Model Training
    # ══════════════════════════════════════════════════════════
    print()
    results = train_models(
        xy_stats=features["xy_stats"],
        xy_latent=features["xy_latent"],
    )

    # Plot results
    plot_results_comparison(results["combined_results"])
    plot_feature_importance(results["importance_df"])

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
