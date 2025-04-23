"""Creates a STAC Item from an event."""

import os
from typing import List

import pandas as pd
import geopandas as gpd
import s3fs
from rashdf import RasPlanHdf
from rashdf.plan import RasPlanHdfError
from pystac import Asset, MediaType
from pystac import Item
from dotenv import load_dotenv
from pystac import Item

from hecstac.common.logger import initialize_logger
from hecstac.events.ffrd import FFRDEventItem

from pathlib import Path

load_dotenv()
fs = s3fs.S3FileSystem(key=os.getenv("AWS_ACCESS_KEY_ID"), secret=os.getenv("AWS_SECRET_ACCESS_KEY"))


def download_if_s3(fpath: str, dest_dir: str = "tmp_downloads") -> str:
    """."""
    if fpath.startswith("s3://"):
        local_dir = Path(dest_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / Path(fpath).name

        fs.get(fpath, str(local_path))

        # Return absolute path
        return str(local_path.resolve())

    return str(Path(fpath).resolve())

if __name__ == "__main__":
    initialize_logger()

    # HMS Info
    # hms_source_model_item_path = "C:\\Users\\sjanke\\Code\\hecstac\\Trinity_1203_EFT_RayRoberts\\Trinity_1203_EFT_RayRoberts.json"
    # hms_source_model_item = Item.from_file(hms_source_model_item_path)

    # RAS Info
    ras_source_model_item_path = (
        "C:\\Users\\sjanke\\Code\\hecstac\\Trinity_1203_EFT_RayRoberts\\Trinity_1203_EFT_RayRoberts.json"
    )
    ras_source_model_item = Item.from_file(ras_source_model_item_path)
    ras_simulation_files = [
        "s3://trinity-pilot/Checkpoint1-ModelsForReview/Hydraulics/EFT-RayRoberts/Model/Trinity_1203_EFT_RayRoberts.p02.hdf"
    ]
    local_ras_simulation_files = [download_if_s3(f) for f in ras_simulation_files]

    # Event Info
    realization = "R01"
    block_group = "BG01"
    event_id = "E01"

    ffrd_event_item_id = f"{realization}-{block_group}-{event_id}"
    dest_href = f"..\\{ffrd_event_item_id}.json"

    ffrd_event_item = FFRDEventItem(
        realization=realization,
        block_group=block_group,
        event_id=event_id,
        source_model_items=[ras_source_model_item],
        ras_simulation_files=local_ras_simulation_files,
    )

    ref_lines_path = "..\\ref_lines.parquet"
    ref_points_path = "..\\ref_points.parquet"
    bc_lines_path = "..\\bc_lines.parquet"

    with (RasPlanHdf(ras_simulation_files) as plan_hdf):
        try:
            plan_hdf.reference_lines().to_parquet(ref_lines_path, engine="pyarrow")
            ffrd_event_item.add_asset("reference_lines", Asset(
                href=ref_lines_path,
                title="Reference Lines",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))
        except RasPlanHdfError as e:
            print(e)

        try:
            plan_hdf.reference_points().to_parquet(ref_points_path, engine="pyarrow")
            ffrd_event_item.add_asset("reference_points", Asset(
                href=ref_points_path,
                title="Reference Points",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))
        except RasPlanHdfError as e:
            print(e)

        try:
            plan_hdf.bc_lines().to_parquet(bc_lines_path, engine="pyarrow")
            ffrd_event_item.add_asset("bc_lines", Asset(
                href=bc_lines_path,
                title="Boundary Condition Lines",
                description="",
                media_type=MediaType.PARQUET,
                roles="rashdf"
            ))
        except RasPlanHdfError as e:
            print(e)

    ffrd_event_item.save_object(dest_href=dest_href)

# reference points, boundary conditions
