import logging
import os
from pathlib import Path

from pystac import Asset, Item


def reorder_stac_assets(stac_item: Item) -> Item:
    if not hasattr(stac_item, "assets") or not isinstance(stac_item.assets, dict):
        raise ValueError("STAC item must have an 'assets' attribute that is a dictionary.")

    data_assets = {k: v for k, v in sorted(stac_item.assets.items()) if hasattr(v, "roles")}  # and ("data" in v.roles)}
    other_assets = {k: v for k, v in sorted(stac_item.assets.items()) if k not in data_assets}
    stac_item.assets = {**data_assets, **other_assets}

    return stac_item


def calibration_plots(stac_item: Item, plot_dir: str) -> Item:
    pngs = Path(plot_dir).rglob("*.png")
    for png in pngs:
        parent_dir = png.parent.parent.name
        asset_title = str(f"{parent_dir}/{png.stem}")
        logging.info(f"Adding asset: {asset_title}")
        new_asset = Asset(href=str(png), title=asset_title, media_type="image/png", roles=["thumbnail"])
        stac_item.add_asset(asset_title, new_asset)
    return stac_item
