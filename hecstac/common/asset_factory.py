"""Create instances of assets."""

from pathlib import Path
from typing import Dict, Generic, Optional, Type, TypeVar

from pyproj import CRS
from pystac import Asset

from hecstac.common.logger import get_logger
from hecstac.hms.s3_utils import check_storage_extension
from hecstac.ras.utils import is_unc_path

T = TypeVar("T")  # Generic for asset file accessor classes


class GenericAsset(Asset, Generic[T]):
    """Provides a base structure for assets."""

    regex_parse_str: str = r""
    __roles__: list[str] = []
    __description__: str = ""
    __media_type__: Optional[str] = None
    __file_class__: T

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.description is None:
            self.description = self.__description__
        self._roles = []
        self._extra_fields = {}
        self.name = Path(self.href).name
        self.media_type = self.__media_type__
        self.logger = get_logger(__file__)

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
    def file(self) -> T:
        """Return cached file or instantiate new."""
        if not hasattr(self, "_file_obj"):
            if self.__file_class__:
                self._file_obj = self.__file_class__(self.href)
            else:
                raise AttributeError(f"No file class defined for {self.__class__.__name__}")
        return self._file_obj

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
        self._cache = {}  # Cache to store created assets

    def create_hms_asset(self, fpath: str, item_type: str = "model") -> Asset:
        """
        Create an asset instance based on the file extension.

        item_type: str

        The type of item to create. This is used to determine the asset class.
        Options are event or model.
        """
        if item_type not in ["event", "model"]:
            raise ValueError(f"Invalid item type: {item_type}, valid options are 'event' or 'model'.")

        # Check cache first
        cache_key = (fpath, item_type)
        if cache_key in self._cache:
            return self._cache[cache_key]

        file_extension = Path(fpath).suffix.lower()
        if file_extension == ".basin":
            asset_class = self.extension_to_asset.get(".basin").get(item_type)
        else:
            asset_class = self.extension_to_asset.get(file_extension, GenericAsset)

        asset = asset_class(href=fpath)
        asset.title = Path(fpath).name

        # Cache the created asset
        self._cache[cache_key] = asset

        return check_storage_extension(asset)

    def asset_from_dict(self, asset: Asset):
        """Create HEC asset given a base Asset and a map of file extensions dict."""
        fpath = asset.href

        # Check cache first
        if fpath in self._cache:
            return self._cache[fpath]

        for pattern, asset_class in self.extension_to_asset.items():
            if pattern.match(fpath):
                # logger.debug(f"Matched {pattern} for {Path(fpath).name}: {asset_class}")
                created_asset = asset_class.from_dict(asset.to_dict())
                # Overwrite the asset href with original filepath if it's a UNC path
                if is_unc_path(fpath):
                    created_asset.href = fpath

                # Cache the created asset
                self._cache[fpath] = created_asset

                return created_asset

        # Cache the generic asset if no match is found
        self._cache[fpath] = asset
        return asset
