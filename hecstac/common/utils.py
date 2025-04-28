"""Common utility functions."""

from pathlib import Path

from pystac import Item


def sanitize_catalog_assets(item: Item) -> Item:
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
