from pathlib import Path
from typing import Dict, Type

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

    def create_asset(self, fpath: str) -> Asset:
        """
        Create an asset instance based on the file extension.
        """
        file_extension = Path(fpath).suffix.lower()
        asset_class = self.extension_to_asset.get(file_extension, GenericAsset)
        asset = asset_class(href=fpath)
        asset.title = Path(fpath).name
        return check_storage_extension(asset)

    def create_asset_from_patter(self, fpath: str) -> Asset:
        # TODO: make this function separate or incorporate in create_asset...dealers choice
        """
        Create an asset instance based on the file extension.
        """
        file_extension = Path(fpath).suffix.lower()
        file_type = ras_file_extensions_type(file_extension)
        asset_class = self.extension_to_asset.get(file_type, GenericAsset)
        asset = asset_class(href=fpath)
        asset.title = Path(fpath).name
        return check_storage_extension(asset)


def ras_file_extensions_type(suffix: str):
    # return ras_file_type
    pass


# RAS_EXTENSION_MAPPING = {
#     ".prj": ProjectAsset,
#     ".p": PlanAsset,
#     ".g": GeometryAsset,
#     ".u": UnsteadyAsset,
# }
