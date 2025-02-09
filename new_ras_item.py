import logging
from pathlib import Path

from hecstac.common.logger import initialize_logger
from hecstac.ras.item import RASModelItem


def sanitize_catalog_assets(item: RASModelItem) -> RASModelItem:
    """
    Forces the asset paths in the catalog relative to item root.
    """
    for asset in item.assets.values():
        if item.pm.model_root_dir in asset.href:
            asset.href = asset.href.replace(item.pm.item_dir, ".")
    return item


if __name__ == "__main__":
    initialize_logger()
    ras_project_file = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.prj"
    item_id = Path(ras_project_file).stem

    ras_item = RASModelItem(ras_project_file, item_id)
    ras_item = sanitize_catalog_assets(ras_item)
    fs = ras_item.scan_model_dir()

    ras_item.add_ras_asset()
    ras_item.save_object(ras_item.pm.item_path(item_id))
    logging.info(f"Saved {ras_item.pm.item_path(item_id)}")
