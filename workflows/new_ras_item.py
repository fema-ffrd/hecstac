"""Creates a STAC Item from a HEC-RAS model .prj file."""

import logging
from pathlib import Path

from hecstac.ras.item import RASModelItem
from hecstac.common.geometry import read_crs_from_prj
from hecstac.common.logger import initialize_logger
from hecstac.common.utils import sanitize_catalog_assets
from rasqc.check import check

if __name__ == "__main__":
    initialize_logger(level=logging.INFO)

    ras_project_file = "s3://trinity-pilot/temp/test/Cedar_1203_CedarCrk.prj"
    # projection_file = "data/rr/Model/Projection.prj"
    # crs = read_crs_from_prj(projection_file)
    crs = None

    item_id = Path(ras_project_file).stem

    ras_item = RASModelItem.from_prj(
        ras_project_file,
        item_id,
        crs=crs,
        assets=[
            "s3://trinity-pilot/temp/test/Cedar_1203_CedarCrk.p01",
            # "s3://trinity-pilot/calibration/hydraulics/clear-fork/clear-fork.g04",
            # "s3://trinity-pilot/calibration/hydraulics/clear-fork/clear-fork.g04.hdf",
        ],
    )

    # ras_item.add_model_thumbnails(["mesh_areas", "breaklines", "bc_lines"])
    ras_item = sanitize_catalog_assets(ras_item)
    item_json = "/workspaces/hecstac/data/clear-fork.json"
    ras_item.save_object(dest_href=item_json)
    results = check(item_json, check_suite="ras_stac_ffrd")
    print(results)
    logging.info(f"Saved {ras_project_file}")
