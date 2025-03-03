"""Features developed duirng R&D for consideration in the hecstac package."""

import logging

from pystac import Item


def add_plan_info(item: Item):
    """Extract plan information from the item assets and add it to the item properties."""
    plans = {}
    for asset_name, asset in item.assets.items():
        if asset.roles:
            if "ras-plan" in asset.roles:
                logging.info(f"Found plan asset: {asset_name}")
                if "HEC-RAS:title" in asset.extra_fields:
                    plans[asset_name] = asset.extra_fields["HEC-RAS:title"].replace("\n", "")
    if len(plans) > 0:
        item.properties["HEC-RAS:plans"] = dict(sorted(plans.items()))
    return item
