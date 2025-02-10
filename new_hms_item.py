"""Creates a STAC Item from a HEC-HMS model .prj file."""

from pathlib import Path

from hecstac.common.logger import initialize_logger
from hecstac import HMSModelItem


def sanitize_catalog_assets(item: HMSModelItem) -> HMSModelItem:
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
    initialize_logger()
    hms_project_file = "duwamish/Duwamish_SST.hms"
    item_id = Path(hms_project_file).stem

    hms_item = HMSModelItem.from_prj(hms_project_file, item_id)
    hms_item = sanitize_catalog_assets(hms_item)
    hms_item.save_object(hms_item.pm.item_path(item_id))
