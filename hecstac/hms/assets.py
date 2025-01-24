from pathlib import Path
from typing import Dict, Type

from pystac import MediaType

from hecstac.common.asset_factory import GenericAsset
from hecstac.hms.parser import (
    BasinFile,
    ControlFile,
    GageFile,
    GridFile,
    MetFile,
    PairedDataFile,
    ProjectFile,
    RunFile,
    SqliteDB,
    TerrainFile,
)
from hecstac.hms.s3_utils import check_storage_extension


class GeojsonAsset(GenericAsset):
    """Geojson asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = [MediaType.GEOJSON]
        description = "Geojson file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class TiffAsset(GenericAsset):
    """Tiff Asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = [MediaType.GEOTIFF]
        description = "Tiff file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class ProjectAsset(GenericAsset):
    """HEC-HMS Project file asset."""

    def __init__(self, href: str, *args, **kwargs):

        roles = ["hms-project"]
        description = "The HEC-HMS project file."

        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.pf = ProjectFile(href, assert_uniform_version=False)
        self.extra_fields = {
            "hms:project_title": self.pf.name,
            "hms:version": self.pf.attrs["Version"],
            "hms:description": self.pf.attrs.get("Description"),
            "hms:unit_system": self.pf.basins[0].header.attrs["Unit System"],
        } | {f"hms:{key}": val for key, val in self.pf.file_counts.items()}


class ThumbnailAsset(GenericAsset):
    """Thumbnail asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["Thumbnail", MediaType.PNG]
        description = "Thumbnail"
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class BasinAsset(GenericAsset):
    """HEC-HMS Basin file asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-basin"]
        description = "Defines the basin geometry and elements for HEC-HMS simulations."
        super().__init__(
            href,
            roles=roles,
            description=description,
            *args,
            **kwargs,
        )
        self.bf = BasinFile(href)
        self.extra_fields = {
            "hms:title": self.bf.name,
            "hms:version": self.bf.header.attrs["Version"],
            "hms:description": self.bf.header.attrs.get("Description"),
            "hms:unit_system": self.bf.header.attrs["Unit System"],
            "hms:gages": self.bf.gages,
            "hms:drainage_area_miles": self.bf.drainage_area,
            "hms:reach_length_miles": self.bf.reach_miles,
            "projection:wkt": self.bf.wkt,
            "projection:code": self.bf.epsg,
        } | {f"hms:{key}".lower(): val for key, val in self.bf.elements.element_counts.items()}


class RunAsset(GenericAsset):
    """Run asset."""

    def __init__(self, href: str, *args, **kwargs):
        self.rf = RunFile(href)
        roles = ["hms-run", MediaType.TEXT]
        description = "Contains data for HEC-HMS simulations."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.extra_fields = {"hms:title": self.name} | {
            run.name: {f"hms:{key}".lower(): val for key, val in run.attrs.items()} for _, run in self.rf.elements
        }


class ControlAsset(GenericAsset):
    """HEC-HMS Control file asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-control", MediaType.TEXT]
        description = "Defines time control information for HEC-HMS simulations."
        super().__init__(
            href,
            roles=roles,
            description=description,
            *args,
            **kwargs,
        )
        self.cf = ControlFile(href)
        self.extra_fields = {
            "hms:title": self.cf.name,
            **{f"hms:{key}".lower(): val for key, val in self.cf.attrs.items()},
        }


class MetAsset(GenericAsset):
    """HEC-HMS Meteorological file asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-met", MediaType.TEXT]
        description = "Contains meteorological data such as precipitation and temperature."
        super().__init__(
            href,
            roles=roles,
            description=description,
            *args,
            **kwargs,
        )
        self.mf = MetFile(href)
        self.extra_fields = {
            "hms:title": self.mf.name,
            **{f"hms:{key}".lower(): val for key, val in self.mf.attrs.items()},
        }


class DSSAsset(GenericAsset):
    """DSS asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hec-dss", "application/octet-stream"]
        description = "HEC-DSS file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)

        self.extra_fields["hms:title"] = self.name


class SqliteAsset(GenericAsset):
    """HEC-HMS SQLite database asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-sqlite", "application/x-sqlite3"]
        description = "Stores spatial data for HEC-HMS basin files."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.sqdb = SqliteDB(href)
        self.extra_fields = {"hms:title": self.name, "hms:layers": self.sqdb.layers}


class GageAsset(GenericAsset):
    """Gage asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-gage", MediaType.TEXT]
        description = "Contains data for HEC-HMS gages."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.gf = GageFile(href)
        self.extra_fields = {"hms:title": self.gf.name, "hms:version": self.gf.attrs["Version"]} | {
            f"hms:{gage.name}".lower(): {key: val for key, val in gage.attrs.items()} for gage in self.gf.gages
        }


class GridAsset(GenericAsset):
    """Grid asset"""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-grid", MediaType.TEXT]
        description = "Contains data for HEC-HMS grid files."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.gf = GridFile(href)
        self.extra_fields = (
            {"hms:title": self.gf.name}
            | {f"hms:{key}".lower(): val for key, val in self.gf.attrs.items()}
            | {f"hms:{grid.name}".lower(): {key: val for key, val in grid.attrs.items()} for grid in self.gf.grids}
        )


class LogAsset(GenericAsset):
    """Log asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-log", MediaType.TEXT]
        description = "Contains log data for HEC-HMS simulations."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.extra_fields["hms:title"] = self.name


class OutAsset(GenericAsset):
    """Out asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-out", MediaType.TEXT]
        description = "Contains output data for HEC-HMS simulations."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.extra_fields["hms:title"] = self.name


class PdataAsset(GenericAsset):
    """Pdata asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-pdata", MediaType.TEXT]
        description = "Contains paired data for HEC-HMS simulations."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.pd = PairedDataFile(href)
        self.extra_fields = {"hms:title": self.pd.name, "hms:version": self.pd.attrs["Version"]}


class TerrainAsset(GenericAsset):
    """Terrain asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hms-terrain"]
        description = "Contains terrain data for HEC-HMS simulations."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.tf = TerrainFile(href)
        self.extra_fields = {"hms:title": self.tf.name, "hms:version": self.tf.attrs["Version"]} | {
            f"hms:{layer['name']}".lower(): {key: val for key, val in layer.items()} for layer in self.tf.layers
        }


HMS_EXTENSION_MAPPING = {
    ".hms": ProjectAsset,
    ".basin": BasinAsset,
    ".control": ControlAsset,
    ".met": MetAsset,
    ".sqlite": SqliteAsset,
    ".gage": GageAsset,
    ".run": RunAsset,
    ".grid": GridAsset,
    ".log": LogAsset,
    ".out": OutAsset,
    ".pdata": PdataAsset,
    ".terrain": TerrainAsset,
    ".dss": DSSAsset,
    ".geojson": GeojsonAsset,
    ".tiff": TiffAsset,
}
