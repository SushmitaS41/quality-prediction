"""Helper functions for the data ingestion pipeline.
"""

import numpy as np
import pandas as pd
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
    - Fill missing timestamps within each Tag interval with `Value = NaN`.
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
        off_grid_adjusted_count = int((grp["Timestamp"] != clipped_ts).sum())
        grp["Timestamp"] = clipped_ts

        # After snapping, duplicates can appear. Keep last observed row.
        grp = grp.drop_duplicates(subset=["Tag", "Timestamp"], keep="last")

        # Build full expected timeline and insert missing timestamps.
        expected_timeline = pd.date_range(start=start_ts, end=end_ts, freq=interval)

        grp = grp.set_index("Timestamp").sort_index()
        grp = grp.reindex(expected_timeline)
        grp["Tag"] = tag
        grp.index.name = "Timestamp"
        grp = grp.reset_index()

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


def outlier_removal(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier values using Isolation Forest.

    Supports two formats:
    - Long format: columns `Tag`, `Value` (fits one model per Tag, removes outlier rows).
    - Wide/pivoted format: columns `fpbatch`, `timestamp`, <sensor1>, <sensor2>, ...
      (fits one model per sensor column, replaces outliers with NaN).
    """

    is_long = {"Tag", "Value"}.issubset(sensor_df.columns)

    contamination = 0.03
    min_points = 10

    if is_long:
        df = sensor_df.copy()
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        keep_mask = pd.Series(True, index=df.index)

        for _tag, idx in df.groupby("Tag", dropna=False).groups.items():
            grp = df.loc[idx, ["Value"]]
            valid_idx = grp["Value"].dropna().index

            if len(valid_idx) < min_points:
                continue

            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200,
            )
            pred = model.fit_predict(df.loc[valid_idx, ["Value"]])
            inlier_idx = valid_idx[pred == 1]

            grp_values = df.loc[idx, "Value"]
            grp_keep = grp_values.isna() | grp_values.index.isin(inlier_idx)
            keep_mask.loc[idx] = grp_keep

        cleaned_df = df.loc[keep_mask].copy()

        sort_cols = [c for c in ["Tag", "Timestamp"] if c in cleaned_df.columns]
        if sort_cols:
            cleaned_df = cleaned_df.sort_values(sort_cols).reset_index(drop=True)
        else:
            cleaned_df = cleaned_df.reset_index(drop=True)

        return cleaned_df

    else:
        df = sensor_df.copy()
        meta_cols = [c for c in ["fpbatch", "timestamp"] if c in df.columns]
        tag_cols = [c for c in df.columns if c not in meta_cols]

        for col in tag_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            valid_idx = series.dropna().index

            if len(valid_idx) < min_points:
                continue

            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=200,
            )
            pred = model.fit_predict(series.loc[valid_idx].values.reshape(-1, 1))
            outlier_idx = valid_idx[pred == -1]
            df.loc[outlier_idx, col] = np.nan

        return df


def impute_small_gaps(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate internal gaps shorter than 60 minutes.

    Supports two formats:
    - Long format: columns `Tag`, `Timestamp`, `Value`.
    - Wide/pivoted format: columns `fpbatch`, `timestamp`, <sensor1>, <sensor2>, ...
      (interpolates each sensor column per fpbatch).
    """

    is_long = {"Tag", "Timestamp", "Value"}.issubset(sensor_df.columns)

    max_gap = pd.Timedelta(minutes=60)

    if is_long:
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

            interpolated = grp["Value"].interpolate(method="linear", limit_area="inside")
            is_missing = grp["Value"].isna()

            if not is_missing.any():
                imputed_groups.append(grp)
                continue

            run_ids = (is_missing != is_missing.shift(fill_value=False)).cumsum()

            for run_id in run_ids[is_missing].unique():
                run_mask = run_ids == run_id
                run_positions = grp.index[run_mask]
                run_length = len(run_positions)
                gap_duration = run_length * cadence

                left_pos = run_positions.min() - 1
                right_pos = run_positions.max() + 1
                has_left_value = left_pos in grp.index and pd.notna(grp.loc[left_pos, "Value"])
                has_right_value = right_pos in grp.index and pd.notna(grp.loc[right_pos, "Value"])

                if gap_duration < max_gap and has_left_value and has_right_value:
                    grp.loc[run_mask, "Value"] = interpolated.loc[run_mask]
                    grp.loc[run_mask, "is_imputed_upto1h"] = grp.loc[run_mask, "Value"].notna()

            imputed_groups.append(grp)

        return pd.concat(imputed_groups, ignore_index=True)

    else:
        df = sensor_df.copy()
        time_col = "timestamp" if "timestamp" in df.columns else "Timestamp"
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        meta_cols = [c for c in ["fpbatch", time_col] if c in df.columns]
        tag_cols = [c for c in df.columns if c not in meta_cols]

        imputed_groups: list[pd.DataFrame] = []

        for _batch, grp in df.groupby("fpbatch", sort=False):
            grp = grp.sort_values(time_col).copy()
            positive_diffs = grp[time_col].diff().dropna()
            positive_diffs = positive_diffs[positive_diffs > pd.Timedelta(0)]

            if positive_diffs.empty:
                imputed_groups.append(grp)
                continue

            cadence = positive_diffs.mode().iloc[0]
            if cadence <= pd.Timedelta(0):
                imputed_groups.append(grp)
                continue

            max_limit = int(max_gap / cadence)

            for col in tag_cols:
                grp[col] = pd.to_numeric(grp[col], errors="coerce")
                grp[col] = grp[col].interpolate(
                    method="linear", limit=max_limit, limit_area="inside"
                )

            imputed_groups.append(grp)

        return pd.concat(imputed_groups, ignore_index=True)


def offset_correction(sensor_df: pd.DataFrame, offset_df: pd.DataFrame) -> pd.DataFrame:
    """Apply sensor-specific timestamp offsets.

    Expected offset columns match the workbook used in the notebooks:
    - `Tag_Names_in_Software`
    - `Offset_Timings_in_seconds`

    Output columns added/updated:
    - `Offset_Seconds`
    - `Timestamp_Adjusted`
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
        return df

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
