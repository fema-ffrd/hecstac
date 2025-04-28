"""Features developed duirng R&D for consideration in the hecstac package."""

from pathlib import Path

from pystac import Asset, Item

from hecstac.common.logger import get_logger


def reorder_stac_assets(stac_item: Item) -> Item:
    """Sort assets alphabetically by key and move data assets to the top."""
    if not hasattr(stac_item, "assets") or not isinstance(stac_item.assets, dict):
        raise ValueError("STAC item must have an 'assets' attribute that is a dictionary.")

    data_assets = {k: v for k, v in sorted(stac_item.assets.items()) if hasattr(v, "roles") and ("data" in v.roles)}
    other_assets = {k: v for k, v in sorted(stac_item.assets.items()) if k not in data_assets}
    stac_item.assets = {**data_assets, **other_assets}

    return stac_item


def calibration_plots(stac_item: Item, plot_dir: str) -> Item:
    """Add existing calibration plots to a STAC item."""
    logger = get_logger(__name__)
    pngs = Path(plot_dir).rglob("*.png")
    for png in pngs:
        parent_dir = png.parent.parent.name
        asset_title = str(f"{parent_dir}/{png.stem}")
        logger.info(f"Adding asset: {asset_title}")
        new_asset = Asset(href=str(png), title=asset_title, media_type="image/png", roles=["thumbnail"])
        stac_item.add_asset(asset_title, new_asset)
    return stac_item
