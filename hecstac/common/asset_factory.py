"""Create instances of assets."""

import logging
from pathlib import Path
from typing import Dict, Type

import pystac
from pyproj import CRS
from pystac import Asset

from hecstac.hms.s3_utils import check_storage_extension

logger = logging.getLogger(__name__)


def is_ras_prj(url: str) -> bool:
    """Check if a file is a HEC-RAS project file."""
    with open(url) as f:
        file_str = f.read()
    if "Proj Title" in file_str.split("\n")[0]:
        return True
    else:
        return False


class GenericAsset(Asset):
    """Provides a base structure for assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.description is None:
            self.description = self.__description__
        self._roles = []
        self._extra_fields = {}

    @property
    def roles(self) -> list[str]:
        """Return roles with enforced values."""
        roles = self._roles
        for i in self.__roles__:
            if i not in roles:
                roles.append(i)
        return roles

    @roles.setter
    def roles(self, roles: list):
        self._roles = roles

    @property
    def extra_fields(self):
        """Return extra fields."""
        # boilerplate here, but overwritten in subclasses
        return self._extra_fields

    @extra_fields.setter
    def extra_fields(self, extra_fields: dict):
        """Set user-defined extra fields."""
        self._extra_fields = extra_fields

    @property
    def file(self):
        """Return class to access asset file contents."""
        return self.__file_class__(self.href)

    def name_from_suffix(self, suffix: str) -> str:
        """Generate a name by appending a suffix to the file stem."""
        return f"{self.stem}.{suffix}"

    @property
    def crs(self) -> CRS:
        """Get the authority code for the model CRS."""
        if self.ext.has("proj"):
            wkt2 = self.ext.proj.wkt2
            if wkt2 is None:
                return
            else:
                return CRS(wkt2)

    def __repr__(self):
        """Return string representation of the GenericAsset instance."""
        return f"<{self.__class__.__name__} name={self.name}>"

    def __str__(self):
        """Return string representation of assets name."""
        return f"{self.name}"


class AssetFactory:
    """Factory for creating HEC asset instances based on file extensions."""

    def __init__(self, extension_to_asset: Dict[str, Type[GenericAsset]]):
        """Initialize the AssetFactory with a mapping of file extensions to asset types and metadata."""
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
        """Create an asset instance based on the file extension."""
        logger.debug(f"Creating asset for {fpath}")
        from hecstac.ras.assets import ProjectAsset

        if fpath.lower().endswith(".prj"):
            if is_ras_prj(fpath):
                return ProjectAsset(href=fpath, title=Path(fpath).name)
            else:
                return GenericAsset(href=fpath, title=Path(fpath).name)

        for pattern, asset_class in self.extension_to_asset.items():
            if pattern.match(fpath):
                logger.debug(f"Matched {pattern} for {Path(fpath).name}: {asset_class}")
                return asset_class(href=fpath, title=Path(fpath).name)

        logger.warning(f"Unable to pattern match asset for file {fpath}")
        return GenericAsset(href=fpath, title=Path(fpath).name)

    def asset_from_dict(self, asset: Asset):
        fpath = asset.href
        for pattern, asset_class in self.extension_to_asset.items():
            if pattern.match(fpath):
                logger.debug(f"Matched {pattern} for {Path(fpath).name}: {asset_class}")
                return asset_class.from_dict(asset.to_dict())
