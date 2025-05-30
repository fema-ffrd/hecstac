import re
import os
import pandas as pd
import fsspec
import logging
from rashdf import RasPlanHdf
from hecstac.common.s3_utils import list_keys_regex, init_s3_resources
from hecstac.common.logger import initialize_logger

fs = fsspec.filesystem("s3")
_, s3_client, _ = init_s3_resources()

bucket = "trinity-pilot"
ras_model_name = "blw-elkhart"
prefix = "calibration/hydraulics"


def list_plan_hdfs(bucket: str, prefix: str) -> list:
    """List all plan HDF files in the given S3 prefix."""
    ras_files = list_keys_regex(s3_client=s3_client, bucket=bucket, prefix_includes=prefix)
    ras_files = [f"s3://{bucket}/{f}" for f in ras_files]
    plan_hdf_files = [f for f in ras_files if re.search(r"\.[p]\d{2}\.hdf$", f)]
    if plan_hdf_files:
        return plan_hdf_files
    else:
        raise ValueError(f"No plan hdf files found at bucket: {bucket} and prefix: {prefix} ")


def save_df_as_pq(df: pd.DataFrame, s3_path: str):
    """Save DataFrame to parquet on S3 if the file does not already exist."""
    if not fs.exists(s3_path):
        df.to_parquet(s3_path)
    else:
        logger.info(f"Skipped existing: {s3_path}")


def process_reference_lines(ref_line_ts, event_id):
    """Process and save flow and water surface data for reference lines."""
    for refln_id in ref_line_ts.refln_id.values:
        refln_name = ref_line_ts.refln_name.sel(refln_id=refln_id).item()

        flow_df = ref_line_ts["Flow"].sel(refln_id=refln_id).to_series().reset_index()
        flow_df.columns = ["time", "flow"]

        wsel_df = ref_line_ts["Water Surface"].sel(refln_id=refln_id).to_series().reset_index()
        wsel_df.columns = ["time", "water_surface"]

        base_path = (
            f"s3://{bucket}/stac/prod-support/results/model={ras_model_name}/event={event_id}/ref_id={refln_name}"
        )
        save_df_as_pq(flow_df, f"{base_path}/flow.pq")
        save_df_as_pq(wsel_df, f"{base_path}/wsel.pq")


def process_reference_points(ref_point_ts, event_id):
    """Process and save velocity and water surface data for reference points."""
    for refpt_id in ref_point_ts.refpt_id.values:
        refpt_name = ref_point_ts.refpt_name.sel(refpt_id=refpt_id).item()

        velocity_df = ref_point_ts["Velocity"].sel(refpt_id=refpt_id).to_series().reset_index()
        velocity_df.columns = ["time", "velocity"]

        wsel_df = ref_point_ts["Water Surface"].sel(refpt_id=refpt_id).to_series().reset_index()
        wsel_df.columns = ["time", "water_surface"]

        base_path = (
            f"s3://{bucket}/stac/prod-support/results/model={ras_model_name}/event={event_id}/ref_id={refpt_name}"
        )
        save_df_as_pq(velocity_df, f"{base_path}/velocity.pq")
        save_df_as_pq(wsel_df, f"{base_path}/wsel.pq")


def main():
    ras_prefix = f"{prefix}/{ras_model_name}"
    plan_hdfs = list_plan_hdfs(bucket, ras_prefix)
    logger.info(f"Found {len(plan_hdfs)} plan hdf files in prefix: {ras_prefix}")

    for plan_path in plan_hdfs:
        logger.info(f"Processing plan hdf file: {plan_path}")
        plan_hdf = RasPlanHdf.open_uri(plan_path)
        plan_name = plan_hdf.get_plan_info_attrs()["Plan Name"]
        event_id = plan_name.split("_")[1]
        logger.info(f"Event id: {event_id}")

        ref_line_ts = plan_hdf.reference_lines_timeseries_output()
        ref_point_ts = plan_hdf.reference_points_timeseries_output()

        process_reference_lines(ref_line_ts, event_id)
        process_reference_points(ref_point_ts, event_id)


if __name__ == "__main__":
    logger = initialize_logger()
    main()
