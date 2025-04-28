"""Creates a STAC Item from an event."""

import os
from pathlib import Path

import s3fs
from dotenv import load_dotenv
from pystac import Item

from hecstac.common.logger import initialize_logger
from hecstac.events.ffrd import FFRDEventItem

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
    ffrd_event_item.save_object(dest_href=dest_href)

# reference points, boundary conditions
