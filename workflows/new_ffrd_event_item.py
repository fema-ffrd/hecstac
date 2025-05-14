"""Creates a STAC Item from an event."""

from typing import List, Tuple
import argparse
import logging
import json
import pandas as pd
from rashdf import RasPlanHdf
from rashdf.plan import RasPlanHdfError
from pystac import Asset, MediaType, Item

from hecstac.common.logger import initialize_logger, get_logger
from hecstac.events.ffrd import FFRDEventItem
from hecstac.common.base_io import ModelFileReader

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
    """Function to parse args from the cli"""
    parser = argparse.ArgumentParser(description="Create STAC Item(s) for FFRD Events.")
    parser.add_argument("ras_model_path", type=str, help="Path to a RAS source model item JSON file.")
    parser.add_argument("ras_simulation_path", type=str, help="Path to a RAS simulation plan HDF file.")

    args = parser.parse_args()

    return (args.ras_model_path, args.ras_simulation_path)

def create_ffrd_item(
        destination_path: str,
        ras_source_model_item: Item,
        ras_simulation_file: str,
        model_name: str,
        realization: str,
        block_group: str,
        event_id: str,
        logger: logging.Logger
    ):
    """Creates an FFRD STAC Item and attaches ref line and ref point time series parquets as assets to the item"""
    ffrd_event_item = FFRDEventItem(
        realization=realization,
        block_group=block_group,
        event_id=event_id,
        source_model_items=[ras_source_model_item],
        ras_simulation_files=[ras_simulation_file]
    )

    ref_lines_path = "ref_lines.parquet"
    ref_points_path = "ref_points.parquet"
    with (RasPlanHdf.open_uri(ras_simulation_file) as ras_plan_hdf):
        try:
            df = ref_lines_ts(ras_plan_hdf, model_name)
            df.to_parquet(ref_lines_path, engine="pyarrow")
            ffrd_event_item.add_asset("reference_lines", Asset(
                href=ref_lines_path,
                title="Reference Lines",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))
        except RasPlanHdfError as e:
            logger.warning(e)

        try:
            df = ref_points_ts(ras_plan_hdf, model_name)
            df.to_parquet(ref_points_path, engine="pyarrow")
            ffrd_event_item.add_asset("reference_points", Asset(
                href=ref_points_path,
                title="Reference Points",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))
        except RasPlanHdfError as e:
            logger.warning(e)

    ffrd_event_item.save_object(dest_href=destination_path)

def main():
    """Script entrypoint for creating a new ffrd event stac item"""
    initialize_logger()
    logger = get_logger("main")
    try:
        ras_model_path, ras_simulation_path = parse_args()

        ras_model_dict = json.loads((ModelFileReader(ras_model_path).content))
        ras_source_model_item = Item.from_dict(ras_model_dict)

        # Event Info
        realization = "R01"
        block_group = "BG01"
        event_id = "E01"
        model_name = "RayRoberts"

        dest_href = f"./new_ffrd_event_item/{realization}-{block_group}-{event_id}.json"

        create_ffrd_item(dest_href, ras_source_model_item, ras_simulation_path, model_name, realization, block_group, event_id, logger)
    except Exception as e:
        logger.exception(f"Fatal error during initialization or processing: {e}")

if __name__ == "__main__":
    main()
