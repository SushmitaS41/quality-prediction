"""Part 1 — Data preparation: ingestion → sanity checks → cleaning → split.

Produces:
    train_df, test_df        (cleaned batch_sensor DataFrames)
    batch_lab_merged         (lab data for target extraction)
"""

import os

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
    val_frac: float = 0.20,
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list, list]:
    """Sort by timestamp, split batches chronologically into train/val/test.

    Works on a copy to perform the split, saves the three splits as CSVs
    in a ``data_source`` folder, and returns the original with train and
    test batches removed (i.e. only validation rows remain).

    Returns (remaining_df, train_df, val_df, test_df,
             train_batches, val_batches, test_batches).
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
    n_val = int(n * val_frac)

    train_batches = batch_order[:n_train]
    val_batches = batch_order[n_train:n_train + n_val]
    test_batches = batch_order[n_train + n_val:]

    train_df = df[df["fpbatch"].isin(train_batches)].reset_index(drop=True)
    val_df = df[df["fpbatch"].isin(val_batches)].reset_index(drop=True)
    test_df = df[df["fpbatch"].isin(test_batches)].reset_index(drop=True)

    # ── Save splits to data_source folder ──
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data_source",
        )
    os.makedirs(save_dir, exist_ok=True)

    train_df.to_csv(os.path.join(save_dir, "batch_sensor_train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "batch_sensor_val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "batch_sensor_test.csv"), index=False)
    print(f"Saved splits to {os.path.abspath(save_dir)}/")

    # ── Remove train and test from the original ──
    remaining_df = batch_sensor_clean[
        ~batch_sensor_clean["fpbatch"].isin(train_batches + test_batches)
    ].reset_index(drop=True)

    test_frac = 1 - train_frac - val_frac
    print(f"Chronological {train_frac:.0%}/{val_frac:.0%}/{test_frac:.0%} split:")
    print(f"  Train: {len(train_batches)} batches  ({len(train_df)} rows)  "
          f"{train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"  Val:   {len(val_batches)} batches  ({len(val_df)} rows)  "
          f"{val_df['timestamp'].min()} → {val_df['timestamp'].max()}")
    print(f"  Test:  {len(test_batches)} batches  ({len(test_df)} rows)  "
          f"{test_df['timestamp'].min()} → {test_df['timestamp'].max()}")
    print(f"  Remaining (original − train − test): {remaining_df['fpbatch'].nunique()} batches  "
          f"({len(remaining_df)} rows)")

    return remaining_df, train_df, val_df, test_df, train_batches, val_batches, test_batches


def extract_target(
    batch_lab_merged: pd.DataFrame,
    target_variable: str = "FilteredArea1",
) -> pd.DataFrame:
    """Extract target rows from lab data, keeping all lane × time combinations.

    Returns DataFrame with columns [fpbatch, lane, time, target] —
    one row per (fpbatch, lane, time) measurement.
    """
    lab = batch_lab_merged.copy()
    lab["value"] = pd.to_numeric(lab["value"], errors="coerce")
    sub = lab[lab["variable"] == target_variable].copy()
    sub = sub.dropna(subset=["value"])
    target = (
        sub[["fpbatch", "lane", "time", "value"]]
        .rename(columns={"value": "target"})
        .reset_index(drop=True)
    )
    print(f"Target '{target_variable}': {len(target)} rows across "
          f"{target['fpbatch'].nunique()} batches "
          f"({target['time'].nunique()} time points, {target['lane'].nunique()} lanes)")
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
    remaining_df, train_df, val_df, test_df, train_batches, val_batches, test_batches = (
        chronological_split(batch_sensor_clean, train_frac)
    )

    print()
    target_df = extract_target(
        batch_lab_merged, target_variable,
    )

    return {
        "batch_lab_merged": batch_lab_merged,
        "batch_sensor_clean": batch_sensor_clean,
        "remaining_df": remaining_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "test_batches": test_batches,
        "target_df": target_df,
    }
