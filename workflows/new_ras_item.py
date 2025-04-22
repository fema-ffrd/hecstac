"""Creates a STAC Item from a HEC-RAS model .prj file."""

import logging
from pathlib import Path

from hecstac.ras.item import RASModelItem
from hecstac.common.geometry import read_crs_from_prj
from hecstac.common.logger import initialize_logger
from hecstac.common.utils import sanitize_catalog_assets


if __name__ == "__main__":
    initialize_logger(level=logging.INFO)

    ras_project_file = "C:\\Users\\sjanke\\Code\\hecstac\\ray-roberts\\ray-roberts.prj"
    # projection_file = "C:\\Users\\sjanke\\Code\\hecstac\\Trinity_1203_EFT_RayRoberts\\Projection.prj"

    crs = None  # = read_crs_from_prj(projection_file)
    item_id = Path(ras_project_file).stem

    ras_item = RASModelItem.from_prj(ras_project_file, item_id, crs=crs)
    # ras_item.add_model_thumbnails(["mesh_areas", "breaklines", "bc_lines"])
    ras_item = sanitize_catalog_assets(ras_item)

    ras_item.save_object(ras_project_file)
    logging.info(f"Saved {ras_project_file}")
