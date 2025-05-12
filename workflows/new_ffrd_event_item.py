"""Creates a STAC Item from an event."""

import os
from typing import List, Tuple
from pathlib import Path
import argparse
import s3fs
import pandas as pd
from rashdf import RasPlanHdf
from rashdf.plan import RasPlanHdfError
from pystac import Asset, MediaType, Item
from dotenv import load_dotenv

from hecstac.common.logger import initialize_logger, get_logger
from hecstac.events.ffrd import FFRDEventItem

def download_if_s3(fpath: str) -> str:
    if fpath.startswith("s3://"):
        load_dotenv(override=True)
        fs = s3fs.S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY"))
        local_dir = Path()
        local_path = local_dir / Path(fpath).name
        print(local_path)

        fs.get(fpath, str(local_path))

        # Return absolute path
        return str(local_path.resolve())

    return str(Path(fpath).resolve())

def ref_lines_ts(plan_hdf: RasPlanHdf, model_name: str) -> pd.DataFrame:
    """Extracts ref line time series data from a RasPlanHdf file"""
    rl_ts_data = []
    rl_ts_ds = plan_hdf.reference_lines_timeseries_output()
    for time in rl_ts_ds.time.values:
        for refln_id in rl_ts_ds.refln_id.values:
            flow = rl_ts_ds["Flow"].sel(time=time, refln_id=refln_id).values
            wsel = rl_ts_ds["Water Surface"].sel(time=time, refln_id=refln_id).values
            # at this point flow and wsel are scalar numpy.ndarray

            rl_ts_data.append({
                'id': f"{model_name}-refln-{refln_id}",
                'time': pd.to_datetime(time),
                'flow': flow.item(),
                'water_surface': wsel.item(),
            })
    return pd.DataFrame(rl_ts_data)

def ref_points_ts(plan_hdf: RasPlanHdf, model_name: str) -> pd.DataFrame:
    """Extracts ref point time series data from a RasPlanHdf file"""
    rl_ts_data = []
    rl_ts_ds = plan_hdf.reference_points_timeseries_output()
    for time in rl_ts_ds.time.values:
        for refpt_id in rl_ts_ds.refpt_id.values:
            velocity = rl_ts_ds["Velocity"].sel(time=time, refpt_id=refpt_id).values
            wsel = rl_ts_ds["Water Surface"].sel(time=time, refpt_id=refpt_id).values
            # at this point velocity and wsel are scalar numpy.ndarray

            rl_ts_data.append({
                'id': f"{model_name}-refpt-{refpt_id}",
                'time': pd.to_datetime(time),
                'velocity': velocity.item(),
                'water_surface': wsel.item(),
            })
    return pd.DataFrame(rl_ts_data)

def parse_args() -> Tuple[str, List[str]]:
    parser = argparse.ArgumentParser(description="Create STAC Item(s) for FFRD Events.")
    parser.add_argument("ras_model_path", type=str, help="Path to a RAS source model item JSON file.")
    parser.add_argument("ras_simulation_path", type=str, help="Path to a RAS simulation plan HDF file.")

    args = parser.parse_args()

    return (args.ras_model_path, args.ras_simulation_path)

def create_ffrd_item(
        destination_path: str,
        ras_source_model_item: Item,
        ras_simulation_file: str,
        realization: str,
        block_group: str,
        event_id: str
    ):
    """Creates an FFRD STAC Item and attaches ref line and ref point time series parquets as assets to the item"""
    ffrd_event_item = FFRDEventItem(
        realization=realization,
        block_group=block_group,
        event_id=event_id,
        source_model_items=[ras_source_model_item],
        ras_simulation_files=ras_simulation_file,
    )

    ref_lines_path = "ref_lines.parquet"
    ref_points_path = "ref_points.parquet"
    bc_lines_path = "bc_lines.parquet"

    with (RasPlanHdf(ras_simulation_file) as ras_plan_hdf):
        try:
            df = ref_lines_ts(ras_plan_hdf, "RayRoberts")
            df.to_parquet(ref_lines_path, engine="pyarrow")
            ffrd_event_item.add_asset("reference_lines", Asset(
                href=ref_lines_path,
                title="Reference Lines",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))

            df = ref_points_ts(ras_plan_hdf, "RayRoberts")
            df.to_parquet(ref_points_path, engine="pyarrow")
            ffrd_event_item.add_asset("reference_points", Asset(
                href=ref_points_path,
                title="Reference Points",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))
        except RasPlanHdfError as e:
            print(e)

    ffrd_event_item.save_object(dest_href=destination_path)

if __name__ == "__main__":
    initialize_logger()
    logger = get_logger("main")
    try:
        ras_model_path, ras_simulation_path = parse_args()

        ras_source_model_item = Item.from_file(download_if_s3(ras_model_path))
        ras_simulation_file = download_if_s3(ras_simulation_path)

        # Event Info
        realization = "R01"
        block_group = "BG01"
        event_id = "E01"

        dest_href = f"{realization}-{block_group}-{event_id}.json"

        create_ffrd_item(dest_href, ras_source_model_item, ras_simulation_file, realization, block_group, event_id)
    except Exception as e:
        logger.exception(f"Fatal error during initialization or processing: {e}")
    