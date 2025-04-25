"""Creates a STAC Item from a HEC-HMS model .prj file."""

from pathlib import Path

from hecstac.common.logger import initialize_logger
from hecstac.common.utils import sanitize_catalog_assets
from hecstac.hms.item import HMSModelItem


if __name__ == "__main__":
    initialize_logger()
    hms_project_file = r"s3://bighorn-ble/catalog/authoritative-models/HUC10080012/HUC10080012.hms"
    item_id = Path(hms_project_file).stem

    hms_item = HMSModelItem.from_prj(hms_project_file, item_id, asset_dir="C:\\Users\\sjanke\\Code\\hecstac\\assets")
    # hms_item = sanitize_catalog_assets(hms_item)
    # hms_item.save_object(hms_item.pm.item_path(item_id))
    hms_item.save_object(dest_href="C:\\Users\\sjanke\\Code\\hecstac\\hms-item-s3.json")
