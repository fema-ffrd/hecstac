"""Common utility functions."""

import json
import os
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


def load_config(config_input: str) -> list[dict[str, str]]:
    """
    Load a HECSTAC workflow configuration from a JSON string or a file path.

    This function is used to load configurations for HECSTAC workflows, such as creating RAS STAC items.
    If `config_input` is a path to a local file, the file is read and parsed as JSON.
    If `config_input` is a JSON string, it is parsed directly.
    The configuration must be either a dictionary or a list of dictionaries.

    Args:
        config_input (str): Path to a JSON config file or a JSON string.

    Returns
    -------
        list[dict]: List of configuration dictionaries.

    Example:
        Single config (as string or file):
            {
                "ras_project_path": "...",
                "output_prefix": "..."
            }
        Multiple configs:
            [
                {"ras_project_path": "...", "output_prefix": "..."},
                {"ras_project_path": "...", "output_prefix": "..."}
            ]
    """
    # add option for reading local file
    if os.path.isfile(config_input):
        with open(config_input, "r") as f:
            config_data = json.load(f)
    else:
        config_data = json.loads(config_input)

    # always pass config as a list
    if isinstance(config_data, dict):
        return [config_data]
    elif isinstance(config_data, list):
        return config_data
    else:
        raise ValueError(f"Config must be a JSON object or list of objects not {type(config_data)}: {config_data}")
