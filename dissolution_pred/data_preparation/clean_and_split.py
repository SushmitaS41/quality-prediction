"""Part 1 — Data preparation: ingestion → sanity checks → cleaning → split.

Produces:
    train_df, test_df        (cleaned batch_sensor DataFrames)
    batch_lab_merged         (lab data for target extraction)
"""

import numpy as np
import pandas as pd

from dissolution_pred.data_preparation.ingestion_pipeline import data_embedding_pipeline


def run_ingestion(
    batch_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    offset_df: pd.DataFrame,
    frequency: str = "1min",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the raw-data ingestion pipeline.

    Returns (batch_lab_merged, batch_sensor_merged).
    """
    batch_lab_merged, batch_sensor_merged = data_embedding_pipeline(
        batch_df, sensor_df, lab_df, offset_df, frequency=frequency,
    )
    print(f"Ingestion complete:")
    print(f"  batch_lab_merged:    {batch_lab_merged.shape}")
    print(f"  batch_sensor_merged: {batch_sensor_merged.shape}")
    return batch_lab_merged, batch_sensor_merged


def sanity_check_and_clean(
    batch_sensor_merged: pd.DataFrame,
    col_nan_threshold: float = 0.90,
) -> pd.DataFrame:
    """Drop columns with >threshold NaN and rows that are 100% NaN on sensors.

    Returns cleaned DataFrame.
    """
    sensor_cols = [c for c in batch_sensor_merged.columns if c.startswith("sensor_")]

    # --- Drop columns with too many NaNs ---
    col_nan_pct = batch_sensor_merged[sensor_cols].isnull().mean()
    drop_cols = col_nan_pct[col_nan_pct > col_nan_threshold].index.tolist()
    df = batch_sensor_merged.drop(columns=drop_cols)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    print(f"Dropped {len(drop_cols)} columns with >{col_nan_threshold*100:.0f}% NaN: {drop_cols}")

    # --- Drop rows that are 100% NaN across sensor columns ---
    all_nan_mask = df[sensor_cols].isnull().all(axis=1)
    n_drop_rows = all_nan_mask.sum()
    df = df[~all_nan_mask].reset_index(drop=True)
    print(f"Dropped {n_drop_rows} rows with 100% missing sensor data")

    print(f"Result: {df.shape}  |  batches: {df['fpbatch'].nunique()}  |  "
          f"remaining NaN: {df[sensor_cols].isnull().sum().sum()}")
    return df


def chronological_split(
    batch_sensor_clean: pd.DataFrame,
    train_frac: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """Sort by timestamp, split batches chronologically.

    Returns (train_df, test_df, train_batches, test_batches).
    """
    df = batch_sensor_clean.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    batch_order = (
        df.groupby("fpbatch")["timestamp"]
        .min()
        .sort_values()
        .index.tolist()
    )

    n = len(batch_order)
    n_train = int(n * train_frac)

    train_batches = batch_order[:n_train]
    test_batches = batch_order[n_train:]

    train_df = df[df["fpbatch"].isin(train_batches)].reset_index(drop=True)
    test_df = df[df["fpbatch"].isin(test_batches)].reset_index(drop=True)

    print(f"Chronological {train_frac:.0%}/{1-train_frac:.0%} split:")
    print(f"  Train: {len(train_batches)} batches  ({len(train_df)} rows)  "
          f"{train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"  Test:  {len(test_batches)} batches  ({len(test_df)} rows)  "
          f"{test_df['timestamp'].min()} → {test_df['timestamp'].max()}")
    return train_df, test_df, train_batches, test_batches


def extract_target(
    batch_lab_merged: pd.DataFrame,
    target_variable: str = "FilteredArea1",
) -> pd.DataFrame:
    """Extract the target column from lab data: mean value per batch.

    Averages across all lanes and time points for the given variable.
    Returns DataFrame with columns [fpbatch, target].
    """
    lab = batch_lab_merged.copy()
    lab["value"] = pd.to_numeric(lab["value"], errors="coerce")
    sub = lab[lab["variable"] == target_variable]
    target = (
        sub.groupby("fpbatch")["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "target"})
    )
    print(f"Target '{target_variable}': {len(target)} batches with values")
    return target


def prepare_data(
    batch_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    offset_df: pd.DataFrame,
    frequency: str = "1min",
    col_nan_threshold: float = 0.90,
    train_frac: float = 0.60,
    target_variable: str = "FilteredArea1",
) -> dict:
    """Full Part 1 pipeline: ingest → clean → split → extract target.

    Returns a dict with all outputs.
    """
    print("=" * 60)
    print("PART 1: DATA PREPARATION")
    print("=" * 60)

    batch_lab_merged, batch_sensor_merged = run_ingestion(
        batch_df, sensor_df, lab_df, offset_df, frequency,
    )

    print()
    batch_sensor_clean = sanity_check_and_clean(
        batch_sensor_merged, col_nan_threshold,
    )

    print()
    train_df, test_df, train_batches, test_batches = chronological_split(
        batch_sensor_clean, train_frac,
    )

    print()
    target_df = extract_target(
        batch_lab_merged, target_variable,
    )

    return {
        "batch_lab_merged": batch_lab_merged,
        "batch_sensor_clean": batch_sensor_clean,
        "train_df": train_df,
        "test_df": test_df,
        "train_batches": train_batches,
        "test_batches": test_batches,
        "target_df": target_df,
    }
