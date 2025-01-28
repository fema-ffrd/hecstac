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


crs = 'PROJCS["NAD_1983_StatePlane_California_III_FIPS_0403_Feet",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199432955]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",6561666.666666666],PARAMETER["False_Northing",1640416.666666667],PARAMETER["Central_Meridian",-120.5],PARAMETER["Standard_Parallel_1",37.06666666666667],PARAMETER["Standard_Parallel_2",38.43333333333333],PARAMETER["Latitude_Of_Origin",36.5],UNIT["Foot_US",0.304800609601219241]]'

if __name__ == "__main__":
    initialize_logger(logging.DEBUG)
    ras_project_file = "Baxter/Baxter.prj"
    item_id = Path(ras_project_file).stem

    ras_item = RASModelItem(ras_project_file, item_id, crs)
    ras_item = sanitize_catalog_assets(ras_item)
    # ras_item.add_model_thumbnail(["mesh_areas", "breaklines", "bc_lines"])
    fs = ras_item.scan_model_dir()

    ras_item.add_ras_asset()
    ras_item.save_object(ras_item.pm.item_path(item_id))
    logging.info(f"Saved {ras_item.pm.item_path(item_id)}")
