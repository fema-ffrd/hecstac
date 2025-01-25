from pathlib import Path

from hecstac.common.logger import initialize_logger
from hecstac.hms.item import HMSModelItem


def sanitize_catalog_assets(item: HMSModelItem) -> HMSModelItem:
    """
    Forces the asset paths in the catalog relative to item root.
    """
    for asset in item.assets.values():
        if item.pm.model_root_dir in asset.href:
            asset.href = asset.href.replace(item.pm.item_dir, ".")
    return item


if __name__ == "__main__":
    initialize_logger()
    hms_project_file = "/Users/slawler/Downloads/duwamish/Duwamish_SST.hms"
    item_id = Path(hms_project_file).stem

    hms_item = HMSModelItem(hms_project_file, item_id)
    hms_item = sanitize_catalog_assets(hms_item)
    hms_item.save_object(hms_item.pm.item_path(item_id))
