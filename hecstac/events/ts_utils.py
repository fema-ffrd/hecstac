"""Utilities to extract specific timeseries from HEC-RAS."""

import os

import pandas as pd
from rashdf import RasPlanHdf
from rashdf.plan import RasPlanHdfError


def save_df_as_pq(df: pd.DataFrame, path: str):
    """Save DataFrame to parquet on local or on s3."""
    if "s3://" in path:
        df.to_parquet(path)
    else:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        df.to_parquet(path)


def save_reference_lines(plan_hdf: RasPlanHdf, output_prefix: str) -> dict:
    """Process and save flow and water surface time series data for reference lines."""
    try:
        ref_line_ts = plan_hdf.reference_lines_timeseries_output()
    except RasPlanHdfError:
        return {}

    refln_paths = {}

    for refln_id in ref_line_ts.refln_id.values:
        refln_name = ref_line_ts.refln_name.sel(refln_id=refln_id).item()

        flow_df = ref_line_ts["Flow"].sel(refln_id=refln_id).to_series().reset_index()
        flow_df.columns = ["time", "flow"]

        wsel_df = ref_line_ts["Water Surface"].sel(refln_id=refln_id).to_series().reset_index()
        wsel_df.columns = ["time", "water_surface"]

        flow_path = f"{output_prefix}/ref_line={refln_name}/flow.pq"
        wsel_path = f"{output_prefix}/ref_line={refln_name}/wsel.pq"

        save_df_as_pq(flow_df, flow_path)
        save_df_as_pq(wsel_df, wsel_path)

        refln_paths[refln_name] = [flow_path, wsel_path]

    return refln_paths


def save_reference_points(plan_hdf: RasPlanHdf, output_prefix: str) -> dict:
    """Process and save velocity and water surface time series data for reference points."""
    try:
        ref_point_ts = plan_hdf.reference_points_timeseries_output()
    except RasPlanHdfError:
        return {}

    refpt_paths = {}

    for refpt_id in ref_point_ts.refpt_id.values:
        refpt_name = ref_point_ts.refpt_name.sel(refpt_id=refpt_id).item()

        velocity_df = ref_point_ts["Velocity"].sel(refpt_id=refpt_id).to_series().reset_index()
        velocity_df.columns = ["time", "velocity"]

        wsel_df = ref_point_ts["Water Surface"].sel(refpt_id=refpt_id).to_series().reset_index()
        wsel_df.columns = ["time", "water_surface"]

        velocity_path = f"{output_prefix}/ref_point={refpt_name}/velocity.pq"
        wsel_path = f"{output_prefix}/ref_point={refpt_name}/wsel.pq"

        save_df_as_pq(velocity_df, velocity_path)
        save_df_as_pq(wsel_df, wsel_path)

        refpt_paths[refpt_name] = [velocity_path, wsel_path]

    return refpt_paths


def save_bc_lines(plan_hdf: RasPlanHdf, output_prefix: str) -> dict:
    """Process and save velocity and water surface time series data for reference points."""
    try:
        bs_line_ts = plan_hdf.bc_lines_timeseries_output()
    except RasPlanHdfError:
        return {}

    bc_ln_paths = {}

    for bc_line_id in bs_line_ts.bc_line_id.values:
        bc_line_name = bs_line_ts.bc_line_name.sel(bc_line_id=bc_line_id).item()

        stage_df = bs_line_ts["Stage"].sel(bc_line_id=bc_line_id).to_series().reset_index()
        stage_df.columns = ["time", "stage"]

        flow_df = bs_line_ts["Flow"].sel(bc_line_id=bc_line_id).to_series().reset_index()
        flow_df.columns = ["time", "flow"]

        stage_path = f"{output_prefix}/bc_line={bc_line_name}/stage.pq"
        flow_path = f"{output_prefix}/bc_line={bc_line_name}/flow.pq"

        save_df_as_pq(stage_df, stage_path)
        save_df_as_pq(flow_df, flow_path)

        bc_ln_paths[bc_line_name] = [stage_path, flow_path]

    return bc_ln_paths
