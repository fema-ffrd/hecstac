"""Creates a STAC Item from a HEC-HMS model .prj file."""

from pathlib import Path

from hecstac.common.logger import initialize_logger
from hecstac.common.utils import sanitize_catalog_assets
from hecstac.hms.item import HMSModelItem


if __name__ == "__main__":
    initialize_logger()
    hms_project_file = r"s3://trinity-pilot/conformance/hydrology/trinity/trinity.hms"
    item_id = Path(hms_project_file).stem

    hms_item = HMSModelItem.from_prj(hms_project_file, item_id, asset_dir="assets")
    # hms_item = sanitize_catalog_assets(hms_item)

    hms_item.save_object(dest_href="<path_to_save_item>/item.json")
