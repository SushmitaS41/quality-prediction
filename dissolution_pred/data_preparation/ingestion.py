import pandas as pd

from utils.ingestion_helpers import (
    impute_small_gaps,
    merge_batch_and_lab,
    merge_sensor_and_batch,
    offset_correction,
    outlier_removal,
    timestamp_sanity_check,
)


def data_ingestion_pipeline(
    batch_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    lab_df: pd.DataFrame,
    offset_df: pd.DataFrame,
    frequency: str = "1min",
) -> pd.DataFrame:
    """Run the end-to-end ingestion pipeline.

    Steps:
    1) Sensor timestamp sanity checks
    2) Sensor outlier removal
    3) Sensor small-gap imputation
    4) Sensor offset correction
    5) Batch-Lab merge
    6) Merge corrected sensor data with Batch-Lab data
    """

    sensor_df = timestamp_sanity_check(sensor_df, frequency=frequency)
    sensor_df = outlier_removal(sensor_df)
    sensor_df = impute_small_gaps(sensor_df)
    sensor_df = offset_correction(sensor_df, offset_df)
    batch_lab_merged = merge_batch_and_lab(batch_df, lab_df)
    batch_sensor_merged = merge_sensor_and_batch(batch_lab_merged, sensor_df)

    return batch_lab_merged, batch_sensor_merged

