"""Creates a STAC Item from a HEC-RAS model .prj file."""

import logging
from pathlib import Path

from hecstac import RASModelItem
from hecstac.common.geometry import read_crs_from_prj
from hecstac.common.logger import initialize_logger
from hecstac.common.proposed_features import calibration_plots, reorder_stac_assets
from hecstac.ras.proposed_features import add_plan_info

# logger = logging.getLogger(__name__)


def sanitize_catalog_assets(item: RASModelItem) -> RASModelItem:
    """Force the asset paths in the catalog to be relative to the item root."""
    item_dir = Path(item.pm.item_dir).resolve()

    for _, asset in item.assets.items():
        asset_path = Path(asset.href).resolve()

        if asset_path.is_relative_to(item_dir):
            asset.href = str(asset_path.relative_to(item_dir))
        else:
            asset.href = (
                str(asset_path.relative_to(item_dir.parent))
                if item_dir.parent in asset_path.parents
                else str(asset_path)
            )

    return item


if __name__ == "__main__":
    initialize_logger(level=logging.INFO)

    ras_project_file = "data/rr/Model/Trinity_1203_EFT_RayRoberts.prj"
    # ras_project_file = "data/cc/HECRAS/Cedar_1203_CedarCrk.prj"
    # ras_project_file = "data/cc/Post-Investigations/CedarCreek/Cedar_1203_CedarCrk.prj"
    projection_file = "data/rr/Model/Projection.prj"
    crs = read_crs_from_prj(projection_file)
    # crs = None

    item_id = Path(ras_project_file).stem

    ras_item = RASModelItem.from_prj(ras_project_file, item_id, crs=crs)
    ras_item.add_model_thumbnails(["mesh_areas", "breaklines", "bc_lines"])
    ras_item = reorder_stac_assets(ras_item)
    # ras_item = calibration_plots(ras_item, "data/cc/Post-Investigations/CedarCreek")
    ras_item = calibration_plots(ras_item, "data/rr/Model/post-investigations/Supporting Documents/Calibration")
    ras_item = add_plan_info(ras_item)
    ras_item = sanitize_catalog_assets(ras_item)

    ras_item.save_object(ras_project_file)
    logging.info(f"Saved {ras_project_file}")
