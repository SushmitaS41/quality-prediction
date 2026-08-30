"""Helper functions for the data ingestion pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest


def timestamp_sanity_check(
    sensor_df: pd.DataFrame,
    frequency: str | pd.Timedelta,
) -> pd.DataFrame:
    """Validate and normalize sensor timestamps by sensor Tag.

    Rules implemented:
    - Cast `Timestamp` to datetime and `Value` to numeric.
    - Group by `Tag` and use caller-provided expected interval.
    - Snap off-grid timestamps to the nearest expected timestamp.
    - Remove duplicates after snapping.
    """

    required_cols = {"Tag", "Timestamp", "Value"}
    missing_cols = required_cols - set(sensor_df.columns)
    if missing_cols:
        raise KeyError(
            f"sensor_df is missing required columns: {sorted(missing_cols)}"
        )

    df = sensor_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    interval = pd.to_timedelta(frequency)

    if interval <= pd.Timedelta(0):
        raise ValueError("frequency must be a positive pandas-compatible timedelta")

    # Rows with missing Tag or invalid Timestamp cannot be placed on a timeline.
    df = df.dropna(subset=["Tag", "Timestamp"]).sort_values(["Tag", "Timestamp"])

    cleaned_groups: list[pd.DataFrame] = []

    for tag, grp in df.groupby("Tag", sort=False):
        grp = grp.sort_values("Timestamp").copy()
        grp = grp.drop_duplicates(subset=["Tag", "Timestamp"], keep="last")

        ts = grp["Timestamp"]
        start_ts = ts.min()
        end_ts = ts.max()
        step_ns = interval.value

        # Snap timestamps to nearest expected cadence anchored at start_ts.
        offset_ns = (grp["Timestamp"] - start_ts).dt.total_seconds() * 1_000_000_000
        snapped_step = np.rint(offset_ns / step_ns).astype("int64")
        snapped_ts = start_ts + pd.to_timedelta(snapped_step * step_ns, unit="ns")

        clipped_ts = snapped_ts.clip(lower=start_ts, upper=end_ts)
        grp["Timestamp"] = clipped_ts

        # After snapping, duplicates can appear. Keep last observed row.
        grp = grp.drop_duplicates(subset=["Tag", "Timestamp"], keep="last")

        # Keep input column order where possible.
        ordered_cols = [c for c in sensor_df.columns if c in grp.columns]
        extra_cols = [c for c in grp.columns if c not in ordered_cols]
        grp = grp[ordered_cols + extra_cols]

        cleaned_groups.append(grp)

    if not cleaned_groups:
        return df.iloc[0:0].copy()

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    cleaned_df = cleaned_df.sort_values(["Tag", "Timestamp"]).reset_index(drop=True)
    return cleaned_df


def outlier_removal(
    sensor_df: pd.DataFrame,
    contamination: float = 0.03,
    min_points: int = 30,
) -> pd.DataFrame:
    """Remove outlier values using Isolation Forest per tag.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Long format with columns `Tag`, `Value`, `Timestamp`.
    contamination : float
        Expected proportion of outliers (default 0.03).
    min_points : int
        Minimum valid observations per tag to run IF.
    """

    if not {"Tag", "Value"}.issubset(sensor_df.columns):
        raise ValueError("Expected long format with columns: Tag, Value.")

    df = sensor_df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values(["Tag", "Timestamp"]).reset_index(drop=True)

    keep_mask = pd.Series(True, index=df.index)

    for _tag, idx in df.groupby("Tag", dropna=False).groups.items():
        values = df.loc[idx, "Value"]
        valid_mask = values.notna()
        valid_values = values[valid_mask]

        if len(valid_values) < min_points:
            continue

        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(valid_values.values.reshape(-1, 1))
        outlier_mask = preds == -1
        keep_mask.loc[valid_values.index[outlier_mask]] = False

    cleaned_df = df.loc[keep_mask].copy()

    sort_cols = [c for c in ["Tag", "Timestamp"] if c in cleaned_df.columns]
    if sort_cols:
        cleaned_df = cleaned_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df


def impute_small_gaps(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate internal gaps shorter than 60 minutes.

    Expects long format: columns `Tag`, `Timestamp`, `Value`.
    """

    max_gap = pd.Timedelta(minutes=60)

    df = sensor_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.sort_values(["Tag", "Timestamp"]).reset_index(drop=True)
    df["is_imputed_upto1h"] = False

    imputed_groups: list[pd.DataFrame] = []

    for _tag, grp in df.groupby("Tag", sort=False):
        grp = grp.sort_values("Timestamp").copy()
        positive_diffs = grp["Timestamp"].diff().dropna()
        positive_diffs = positive_diffs[positive_diffs > pd.Timedelta(0)]

        if positive_diffs.empty:
            imputed_groups.append(grp)
            continue

        cadence = positive_diffs.mode().iloc[0]
        if cadence <= pd.Timedelta(0):
            imputed_groups.append(grp)
            continue

        is_missing = grp["Value"].isna()
        if not is_missing.any():
            imputed_groups.append(grp)
            continue

        max_limit = int(max_gap / cadence) - 1  # strict < 60 min
        if max_limit < 1:
            imputed_groups.append(grp)
            continue

        grp["Value"] = grp["Value"].interpolate(
            method="linear", limit=max_limit, limit_area="inside"
        )
        grp["is_imputed_upto1h"] = is_missing & grp["Value"].notna()

        imputed_groups.append(grp)

    return pd.concat(imputed_groups, ignore_index=True)


def impute_all_gaps(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate all internal NaN gaps per tag, no time limit.

    Expects long format: columns `Tag`, `Timestamp`, `Value`.
    Adds `is_imputed` flag column for filled rows.
    """

    df = sensor_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.sort_values(["Tag", "Timestamp"]).reset_index(drop=True)
    df["is_imputed"] = False

    for tag, idx in df.groupby("Tag", sort=False).groups.items():
        values = df.loc[idx, "Value"]
        is_missing = values.isna()
        if not is_missing.any():
            continue
        interpolated = values.interpolate(method="linear", limit_area="inside")
        filled_mask = is_missing & interpolated.notna()
        df.loc[idx, "Value"] = interpolated
        df.loc[idx[filled_mask], "is_imputed"] = True

    return df


def offset_correction(
    sensor_df: pd.DataFrame,
    offset_df: pd.DataFrame,
    align_to_grid: bool = False,
    align_frequency: str | pd.Timedelta = "1min",
    align_tolerance: str | pd.Timedelta = "30s",
) -> pd.DataFrame:
    """Apply sensor-specific timestamp offsets.

    Expected offset columns match the workbook used in the notebooks:
    - `Tag_Names_in_Software`
    - `Offset_Timings_in_seconds`

    Output columns added/updated:
    - `Offset_Seconds`
    - `Timestamp_Adjusted`

        Optional alignment:
        - If `align_to_grid=True`, adds `Timestamp_Aligned` by snapping
            `Timestamp_Adjusted` to nearest `align_frequency` grid point
            only when delta is within `align_tolerance`.
        - Adds `Aligned_Within_Tolerance` flag when alignment is enabled.
    """

    required_sensor_cols = {"Tag", "Timestamp"}
    missing_sensor_cols = required_sensor_cols - set(sensor_df.columns)
    if missing_sensor_cols:
        raise KeyError(
            f"sensor_df is missing required columns: {sorted(missing_sensor_cols)}"
        )

    df = sensor_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    if offset_df.empty:
        df["Offset_Seconds"] = 0.0
        df["Timestamp_Adjusted"] = df["Timestamp"]
    else:
        required_offset_cols = {"Tag_Names_in_Software", "Offset_Timings_in_seconds"}
        missing_offset_cols = required_offset_cols - set(offset_df.columns)
        if missing_offset_cols:
            raise KeyError(
                f"offset_df is missing required columns: {sorted(missing_offset_cols)}"
            )

        offsets = offset_df[["Tag_Names_in_Software", "Offset_Timings_in_seconds"]].copy()
        offsets["Offset_Timings_in_seconds"] = pd.to_numeric(
            offsets["Offset_Timings_in_seconds"],
            errors="coerce",
        )

        offset_map = dict(
            zip(
                offsets["Tag_Names_in_Software"],
                offsets["Offset_Timings_in_seconds"],
            )
        )

        df["Offset_Seconds"] = df["Tag"].map(offset_map).fillna(0.0)
        df["Timestamp_Adjusted"] = df["Timestamp"] - pd.to_timedelta(
            df["Offset_Seconds"],
            unit="s",
        )

    if align_to_grid:
        frequency = pd.to_timedelta(align_frequency)
        tolerance = pd.to_timedelta(align_tolerance)

        if frequency <= pd.Timedelta(0):
            raise ValueError("align_frequency must be a positive pandas-compatible timedelta")
        if tolerance < pd.Timedelta(0):
            raise ValueError("align_tolerance must be >= 0")

        rounded = df["Timestamp_Adjusted"].dt.round(frequency)
        delta = (df["Timestamp_Adjusted"] - rounded).abs()

        df["Aligned_Within_Tolerance"] = delta <= tolerance
        df["Timestamp_Aligned"] = rounded.where(df["Aligned_Within_Tolerance"], pd.NaT)

    sort_cols = [c for c in ["Tag", "Timestamp_Adjusted", "Timestamp"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def merge_batch_and_lab(batch_df: pd.DataFrame, lab_df: pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    """Merge batch and lab datasets on the shared `fpbatch` key.

    Expected inputs:
    - batch_df: `fpid`, `prodname`, `fpbatch`, `order`, `starttime`, `endtime`
    - lab_df: `fpbatch`, `prodname`, `variable`, `value`, `lane`, `time`

    Behavior:
    - Drops helper/index columns such as `Unnamed: 0` and lab `index`.
    - Normalizes `fpbatch` to stripped string on both inputs.
    - Keeps lab `time` column as-is (no datetime parsing).
    - Preserves both product name columns as `prodname_batch` and `prodname_lab`.
    - Uses an inner join so only overlapping `fpbatch` values are retained.
    """

    required_batch_cols = {"fpbatch"}
    required_lab_cols = {"fpbatch"}

    missing_batch_cols = required_batch_cols - set(batch_df.columns)
    missing_lab_cols = required_lab_cols - set(lab_df.columns)

    if missing_batch_cols:
        raise KeyError(
            f"batch_df is missing required columns: {sorted(missing_batch_cols)}"
        )

    if missing_lab_cols:
        raise KeyError(
            f"lab_df is missing required columns: {sorted(missing_lab_cols)}"
        )

    batch_clean = batch_df.copy()
    lab_clean = lab_df.copy()

    batch_drop_cols = [col for col in ["Unnamed: 0"] if col in batch_clean.columns]
    lab_drop_cols = [col for col in ["Unnamed: 0", "index"] if col in lab_clean.columns]

    if batch_drop_cols:
        batch_clean = batch_clean.drop(columns=batch_drop_cols)

    if lab_drop_cols:
        lab_clean = lab_clean.drop(columns=lab_drop_cols)

    batch_clean["fpbatch"] = batch_clean["fpbatch"].astype("string").str.strip()
    lab_clean["fpbatch"] = lab_clean["fpbatch"].astype("string").str.strip()

    batch_clean = batch_clean[
        batch_clean["fpbatch"].notna() & (batch_clean["fpbatch"] != "")
    ].copy()
    lab_clean = lab_clean[
        lab_clean["fpbatch"].notna() & (lab_clean["fpbatch"] != "")
    ].copy()

    if "starttime" in batch_clean.columns:
        batch_clean["starttime"] = pd.to_datetime(batch_clean["starttime"], errors="coerce")

    if "endtime" in batch_clean.columns:
        batch_clean["endtime"] = pd.to_datetime(batch_clean["endtime"], errors="coerce")

    merged_df = batch_clean.merge(
        lab_clean,
        on="fpbatch",
        how="inner",
        suffixes=("_batch", "_lab"),
    )

    sort_cols = [col for col in ["fpbatch", "variable", "time"] if col in merged_df.columns]
    if sort_cols:
        merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        merged_df = merged_df.reset_index(drop=True)

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "batch_lab_merged.csv")
    merged_df.to_csv(output_path, index=False)

    return merged_df


def merge_sensor_and_batch(
    batch_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    offset_df: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Build a dataset linking each fpbatch to sensor readings within its time window.

    For each unique batch, shifts the batch window per tag using offset_df
    (so sensor timestamps stay on the original 1-minute grid), then pivots
    so each unique Tag becomes its own column.

    Parameters
    ----------
    batch_df : pd.DataFrame
        Must contain columns: `fpbatch`, `starttime`, `endtime`.
    sensor_df : pd.DataFrame
        Must contain columns: `Tag`, `Value`, `Timestamp`.
    offset_df : pd.DataFrame, optional
        Columns: `Tag_Names_in_Software`, `Offset_Timings_in_seconds`.
        If provided, adjusts batch windows per tag instead of shifting sensor timestamps.
    output_path : str, optional
        If provided, saves the result as a CSV to this path.

    Returns
    -------
    pd.DataFrame
        Columns: `fpbatch`, `timestamp`, <Tag1>, <Tag2>, ...
    """

    required_batch_cols = {"fpbatch", "starttime", "endtime"}
    missing_batch_cols = required_batch_cols - set(batch_df.columns)
    if missing_batch_cols:
        raise KeyError(f"batch_df is missing required columns: {sorted(missing_batch_cols)}")

    required_sensor_cols = {"Tag", "Value"}
    missing_sensor_cols = required_sensor_cols - set(sensor_df.columns)
    if missing_sensor_cols:
        raise KeyError(f"sensor_df is missing required columns: {sorted(missing_sensor_cols)}")

    batch = batch_df.copy()
    batch["starttime"] = pd.to_datetime(batch["starttime"], errors="coerce")
    batch["endtime"] = pd.to_datetime(batch["endtime"], errors="coerce")

    sensor = sensor_df.copy()
    sensor["Timestamp"] = pd.to_datetime(sensor["Timestamp"], errors="coerce")
    sensor["Value"] = pd.to_numeric(sensor["Value"], errors="coerce")

    # Build offset map (tag -> seconds) if offset_df is provided
    offset_map: dict[str, float] = {}
    if offset_df is not None and not offset_df.empty:
        req = {"Tag_Names_in_Software", "Offset_Timings_in_seconds"}
        if req.issubset(offset_df.columns):
            off = offset_df[["Tag_Names_in_Software", "Offset_Timings_in_seconds"]].copy()
            off["Offset_Timings_in_seconds"] = pd.to_numeric(off["Offset_Timings_in_seconds"], errors="coerce")
            offset_map = dict(zip(off["Tag_Names_in_Software"], off["Offset_Timings_in_seconds"]))

    all_tags = sensor["Tag"].dropna().unique()

    # Pre-group sensor by Tag once — avoids repeated full-DataFrame scans
    sensor_by_tag: dict[str, pd.DataFrame] = {}
    for tag in all_tags:
        tag_df = sensor.loc[sensor["Tag"] == tag, ["Timestamp", "Value"]].copy()
        tag_df = tag_df.set_index("Timestamp").sort_index()
        sensor_by_tag[tag] = tag_df

    results = []

    for _, batch_row in batch.iterrows():
        fpbatch = batch_row["fpbatch"]
        start = batch_row["starttime"]
        end = batch_row["endtime"]

        if pd.isna(start) or pd.isna(end):
            continue

        tag_slices = []
        for tag in all_tags:
            tag_df = sensor_by_tag[tag]
            offset_s = offset_map.get(tag, 0.0)
            if pd.isna(offset_s):
                offset_s = 0.0
            adj_start = start + pd.Timedelta(seconds=offset_s)
            adj_end = end + pd.Timedelta(seconds=offset_s)

            # Fast slice on sorted DatetimeIndex (strict inequality: > start, < end)
            slc = tag_df[(tag_df.index > adj_start) & (tag_df.index < adj_end)]
            if slc.empty:
                continue
            slc = slc.rename(columns={"Value": tag})
            tag_slices.append(slc)

        if not tag_slices:
            continue

        # Join all tags on the shared Timestamp index (1-minute grid)
        # Use inner join so only timestamps present for ALL tags are kept — no edge NaNs
        merged = tag_slices[0]
        for ts in tag_slices[1:]:
            merged = merged.join(ts, how="inner")

        merged.index.name = "timestamp"
        merged = merged.reset_index()

        merged["fpbatch"] = fpbatch
        results.append(merged)

    if not results:
        return pd.DataFrame(columns=["fpbatch", "timestamp"])

    batch_sensor_df = pd.concat(results, ignore_index=True)

    # Move fpbatch and timestamp to front
    front_cols = ["fpbatch", "timestamp"]
    tag_cols = [c for c in batch_sensor_df.columns if c not in front_cols]
    batch_sensor_df = batch_sensor_df[front_cols + tag_cols]

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "batch_sensor_merged.csv")
    batch_sensor_df.to_csv(output_path, index=False)

    return batch_sensor_df


def plot_outlier_removal(before_df: pd.DataFrame, after_df: pd.DataFrame, save_dir: str = "plots", sample_tags: int = 14):
    """Plot and save outlier removal sanity check."""

    os.makedirs(save_dir, exist_ok=True)

    before = before_df.copy()
    after = after_df.copy()
    before["Value"] = pd.to_numeric(before["Value"], errors="coerce")
    after["Value"] = pd.to_numeric(after["Value"], errors="coerce")
    before["Timestamp"] = pd.to_datetime(before["Timestamp"], errors="coerce")
    after["Timestamp"] = pd.to_datetime(after["Timestamp"], errors="coerce")

    top_tags = before["Tag"].dropna().unique()
    tags_with_removals = []
    for tag in top_tags:
        b_count = before.loc[before["Tag"] == tag, "Value"].notna().sum()
        a_count = after.loc[after["Tag"] == tag, "Value"].notna().sum()
        if b_count > a_count:
            tags_with_removals.append((tag, b_count - a_count))
    tags_with_removals.sort(key=lambda x: x[1], reverse=True)
    tags_to_plot = [t for t, _ in tags_with_removals[:sample_tags]]

    if not tags_to_plot:
        return

    n = len(tags_to_plot)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle("Outlier Removal Sanity Check", fontsize=14, y=1.01)

    for i, tag in enumerate(tags_to_plot):
        b = before[before["Tag"] == tag].sort_values("Timestamp")
        a = after[after["Tag"] == tag].sort_values("Timestamp")
        removed_idx = b.index.difference(a.index)
        removed = b.loc[removed_idx]

        ax1 = axes[i, 0]
        ax1.plot(b["Timestamp"], b["Value"], ".", ms=1, alpha=0.4, label="before")
        ax1.plot(removed["Timestamp"], removed["Value"], "rx", ms=5, label=f"removed ({len(removed)})")
        ax1.set_title(f"{tag} — time series")
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="x", rotation=30)

        ax2 = axes[i, 1]
        ax2.hist(b["Value"].dropna(), bins=50, alpha=0.5, label="before", density=True)
        ax2.hist(a["Value"].dropna(), bins=50, alpha=0.5, label="after", density=True)
        ax2.set_title(f"{tag} — distribution")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "outlier_removal_sanity.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_imputation(before_df: pd.DataFrame, after_df: pd.DataFrame, save_dir: str = "plots", sample_tags: int = 6):
    """Plot and save imputation sanity check using is_imputed_upto1h flag."""

    os.makedirs(save_dir, exist_ok=True)

    after = after_df.copy()
    after["Value"] = pd.to_numeric(after["Value"], errors="coerce")
    after["Timestamp"] = pd.to_datetime(after["Timestamp"], errors="coerce")

    before = before_df.copy()
    before["Value"] = pd.to_numeric(before["Value"], errors="coerce")
    before["Timestamp"] = pd.to_datetime(before["Timestamp"], errors="coerce")

    # Use is_imputed or is_imputed_upto1h flag if available
    flag_col = None
    if "is_imputed" in after.columns:
        flag_col = "is_imputed"
    elif "is_imputed_upto1h" in after.columns:
        flag_col = "is_imputed_upto1h"

    summary = []
    for tag in before["Tag"].dropna().unique():
        b_valid = before.loc[before["Tag"] == tag, "Value"].notna().sum()
        a_valid = after.loc[after["Tag"] == tag, "Value"].notna().sum()
        filled = int(a_valid - b_valid)
        imputed_flag = int(after.loc[after["Tag"] == tag, flag_col].sum()) if flag_col else filled
        summary.append({"Tag": tag, "filled": filled, "imputed_flag": imputed_flag})

    summary.sort(key=lambda x: x["filled"], reverse=True)
    tags_to_plot = [s["Tag"] for s in summary if s["filled"] > 0][:sample_tags]

    if not tags_to_plot:
        return

    n = len(tags_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)
    fig.suptitle("Imputation Sanity Check", fontsize=14)

    for i, tag in enumerate(tags_to_plot):
        ax = axes[i, 0]
        a = after[after["Tag"] == tag].sort_values("Timestamp")

        if flag_col:
            imputed_pts = a[a[flag_col] == True]
        else:
            b = before[before["Tag"] == tag].sort_values("Timestamp")
            imputed_pts = a[b["Value"].isna().values & a["Value"].notna().values]

        ax.plot(a["Timestamp"], a["Value"], ".", ms=1, alpha=0.3, color="steelblue", label="data")
        ax.plot(imputed_pts["Timestamp"], imputed_pts["Value"], ".", ms=4, color="red", alpha=0.7,
                label=f"imputed ({len(imputed_pts)})")
        ax.set_title(f"{tag} — {len(imputed_pts)} points filled, {int(a['Value'].isna().sum())} still NaN")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "imputation_sanity.png"), dpi=150, bbox_inches="tight")
    plt.show()


def batch_feature_stats(sensor_batch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-batch statistical feature vector from wide-format sensor-batch data.

    Expects columns: `fpbatch`, `timestamp`, <sensor1>, <sensor2>, ...

    For each batch and each sensor, computes:
      mean, std, min, max, median, skew, range, pct_missing

    Returns one row per batch with flattened column names like
    `sensor_1_pv_mean`, `sensor_1_pv_pct_missing`, etc.
    """

    df = sensor_batch_df.copy()
    meta_cols = ["fpbatch", "timestamp"]
    sensor_cols = [c for c in df.columns if c not in meta_cols]

    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    records = []
    for fpbatch, grp in df.groupby("fpbatch", sort=False):
        row = {"fpbatch": fpbatch, "n_timestamps": len(grp)}
        for col in sensor_cols:
            vals = grp[col]
            valid = vals.dropna()
            row[f"{col}_mean"] = valid.mean() if len(valid) else np.nan
            row[f"{col}_std"] = valid.std() if len(valid) > 1 else np.nan
            row[f"{col}_min"] = valid.min() if len(valid) else np.nan
            row[f"{col}_max"] = valid.max() if len(valid) else np.nan
            row[f"{col}_median"] = valid.median() if len(valid) else np.nan
            row[f"{col}_skew"] = valid.skew() if len(valid) > 2 else np.nan
            row[f"{col}_range"] = (valid.max() - valid.min()) if len(valid) else np.nan
            row[f"{col}_pct_missing"] = round(100 * vals.isna().sum() / max(len(vals), 1), 2)
        records.append(row)

    result = pd.DataFrame(records)

    # Fill remaining NaN stats with the column median, then drop columns still all-NaN
    stat_cols = [c for c in result.columns if c not in ("fpbatch", "n_timestamps")]
    for col in stat_cols:
        col_median = result[col].median()
        result[col] = result[col].fillna(col_median)
    # If a column is still all-NaN (sensor had zero data everywhere), drop it
    result = result.dropna(axis=1, how="all")

    return result
