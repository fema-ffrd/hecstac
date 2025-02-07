"""Creates a STAC Item from a HEC-RAS model .prj file."""

import logging
from pathlib import Path

from hecstac import RASModelItem
from hecstac.ras.logger import initialize_logger

logger = logging.getLogger(__name__)


def sanitize_catalog_assets(item: RASModelItem) -> RASModelItem:
    """Force the asset paths in the catalog to be relative to the item root."""
    item_dir = Path(item.pm.item_dir).resolve()
    for _, asset in item.assets.items():

        asset_path = Path(asset.href).resolve()

        if asset_path.is_relative_to(item_dir):
            asset.href = str(asset_path.relative_to(item_dir))

        elif asset.href.startswith(f"{item_dir.name}/"):
            asset.href = asset.href.replace(f"{item_dir.name}/", "", 1)

    return item


if __name__ == "__main__":
    initialize_logger()
    ras_project_file = "Example_Projects_6_6/2D Unsteady Flow Hydraulics/Muncie/Muncie.prj"
    item_id = Path(ras_project_file).stem

    ras_item = RASModelItem(ras_project_file, item_id, crs=None)
    ras_item = sanitize_catalog_assets(ras_item)
    # ras_item.add_model_thumbnails(["mesh_areas", "breaklines", "bc_lines"])
    ras_item.save_object(ras_item.pm.item_path(item_id))
    logger.info(f"Saved {ras_item.pm.item_path(item_id)}")
