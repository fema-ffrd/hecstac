import logging
from pathlib import Path
from typing import Dict, Type

import pystac
from pyproj import CRS
from pystac import Asset

from hecstac.hms.s3_utils import check_storage_extension


class GenericAsset(Asset):
    """Generic Asset."""

    def __init__(self, href: str, roles=None, description=None, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.href = href
        self.name = Path(href).name
        self.stem = Path(href).stem
        self.roles = roles or []
        self.description = description or ""

    def name_from_suffix(self, suffix: str) -> str:
        """Generate a name by appending a suffix to the file stem."""
        return f"{self.stem}.{suffix}"

    @property
    def crs(self) -> CRS:
        """Get the authority code for the model CRS."""
        try:
            wkt2 = self.ext.proj.wkt2
            if wkt2 is None:
                return
            else:
                return CRS(wkt2)
        except pystac.errors.ExtensionNotImplemented:
            return None

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"

    def __str__(self):
        return f"{self.name}"


class AssetFactory:
    """Factory for creating HEC asset instances based on file extensions."""

    def __init__(self, extension_to_asset: Dict[str, Type[GenericAsset]]):
        """
        Initialize the AssetFactory with a mapping of file extensions to asset types and metadata.
        """
        self.extension_to_asset = extension_to_asset

    def create_hms_asset(self, fpath: str, item_type: str = "model") -> Asset:
        """
        Create an asset instance based on the file extension.
        item_type: str

        The type of item to create. This is used to determine the asset class.
        Options are event or model.
        """
        if item_type not in ["event", "model"]:
            raise ValueError(f"Invalid item type: {item_type}, valid options are 'event' or 'model'.")

        file_extension = Path(fpath).suffix.lower()
        if file_extension == ".basin":
            asset_class = self.extension_to_asset.get(".basin").get(item_type)
        else:
            asset_class = self.extension_to_asset.get(file_extension, GenericAsset)

        asset = asset_class(href=fpath)
        asset.title = Path(fpath).name
        return check_storage_extension(asset)

    def create_ras_asset(self, fpath: str):
        logging.debug(f"Creating asset for {fpath}")
        for pattern, asset_class in self.extension_to_asset.items():
            if pattern.match(fpath):
                logging.debug(f"Matched {pattern} for {Path(fpath).name}: {asset_class}")
                return asset_class(href=fpath, title=Path(fpath).name)

        logging.warning(f"Unable to pattern match asset for file {fpath}")
        return GenericAsset(href=fpath, title=Path(fpath).name)

    def asset_from_dict(self, asset: Asset):
        fpath = asset.href
        for pattern, asset_class in self.extension_to_asset.items():
            if pattern.match(fpath):
                logging.debug(f"Matched {pattern} for {Path(fpath).name}: {asset_class}")
                return asset_class.from_dict(asset.to_dict())
