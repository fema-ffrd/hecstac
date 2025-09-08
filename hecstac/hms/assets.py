"""HEC-HMS Stac Item asset classes."""

from pystac import MediaType
import re
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

HMS_VERSION = "hms:version"
HMS_TITLE = "hms:title"
HMS_DESCRIPTION = "hms:description"
HMS_UNIT_SYSTEM = "hms:unit_system"
HMS_BASIN_TITLE = "hms:basin_title"


class GeojsonAsset(GenericAsset):
    """Geojson asset."""

    regex_parse_str = r".*\.geojson$"
    __roles__ = ["data"]
    __description__ = "Geojson file."
    __media_type__ = MediaType.GEOJSON


class TiffAsset(GenericAsset):
    """Tiff Asset."""

    regex_parse_str = r".*\.tiff$"
    __roles__ = ["data", MediaType.GEOTIFF]
    __description__ = "Tiff file."


class ProjectAsset(GenericAsset[ProjectFile]):
    """Project asset."""

    regex_parse_str = r".*\.hms$"
    __roles__ = ["hms-project", MediaType.TEXT]
    __description__ = "The HEC-HMS project file. Summary provided at the item level"
    __file_class__ = ProjectFile


class ThumbnailAsset(GenericAsset):
    """Thumbnail asset."""

    regex_parse_str = r".*\.png$"
    __roles__ = ["thumbnail", MediaType.PNG]
    __description__ = "Thumbnail"

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        basin_file = self.href.split(".")[0] + ".basin"
        return {"associated_basin_file": basin_file}


class ModelBasinAsset(GenericAsset[BasinFile]):
    """HEC-HMS Basin file asset from authoritative model, containing geometry and other detailed data."""

    regex_parse_str = r".*\.basin$"
    __roles__ = ["hms-basin", MediaType.TEXT]
    __description__ = "Defines the basin geometry and elements for HEC-HMS simulations."
    __file_class__ = BasinFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return (
            {
                HMS_BASIN_TITLE: self.file.name,
                HMS_VERSION: self.file.header.attrs["Version"],
                HMS_DESCRIPTION: self.file.header.attrs.get("Description"),
                HMS_UNIT_SYSTEM: self.file.header.attrs["Unit System"],
                "hms:gages": self.file.gages,
                "hms:drainage_area_miles": self.file.drainage_area,
                "hms:reach_length_miles": self.file.reach_miles,
                "proj:wkt": self.file.wkt,
                "proj:code": self.file.epsg,
            }
            | self.file.hms_methods
            | {f"hms_basin:{key.lower()}": val for key, val in self.file.elements.element_counts.items()}
        )


class EventBasinAsset(GenericAsset[BasinFile]):
    """HEC-HMS Basin file asset from event, with limited basin info."""

    regex_parse_str = r".*\.basin$"
    __roles__ = ["hms-basin", MediaType.TEXT]
    __description__ = "Defines the basin geometry and elements for HEC-HMS simulations."
    __file_class__ = BasinFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {
            HMS_BASIN_TITLE: self.file.name,
            HMS_VERSION: self.file.header.attrs["Version"],
            HMS_DESCRIPTION: self.file.header.attrs.get("Description"),
            HMS_UNIT_SYSTEM: self.file.header.attrs["Unit System"],
        }


class RunAsset(GenericAsset[RunFile]):
    """Run asset."""

    regex_parse_str = r".*\.run$"
    __file_class__ = RunFile
    __roles__ = ["hms-run", MediaType.TEXT]
    __description__ = "Contains data for HEC-HMS simulations."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {"hms:run_title": self.name.removesuffix(".run")} | {
            run.name: {f"hms:{key}".lower(): val for key, val in run.attrs.items()} for _, run in self.file.elements
        }


class ControlAsset(GenericAsset[ControlFile]):
    """HEC-HMS Control file asset."""

    regex_parse_str = r".*\.control$"
    __roles__ = ["hms-control", MediaType.TEXT]
    __description__ = "Defines time control information for HEC-HMS simulations."
    __file_class__ = ControlFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {
            "hms:control_title": self.file.name,
            **{f"hms:{key}".lower(): val for key, val in self.file.attrs.items()},
        }


class MetAsset(GenericAsset[MetFile]):
    """HEC-HMS Meteorological file asset."""

    regex_parse_str = r".*\.met$"
    __roles__ = ["hms-met", MediaType.TEXT]
    __description__ = "Contains meteorological data such as precipitation and temperature."
    __file_class__ = MetFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {
            "hms:met_title": self.file.name,
            **{f"hms:{key}".lower(): val for key, val in self.file.attrs.items()},
        }


class DSSAsset(GenericAsset):
    """DSS asset."""

    regex_parse_str = r".*\.dss$"
    __roles__ = ["hec-dss", "application/octet-stream"]
    __description__ = "HEC-DSS file."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.name}


class SqliteAsset(GenericAsset[SqliteDB]):
    """HEC-HMS SQLite database asset."""

    regex_parse_str = r".*\.sqlite$"
    __roles__ = ["hms-sqlite", "application/x-sqlite3"]
    __description__ = "Stores spatial data for HEC-HMS basin files."
    __file_class__ = SqliteDB

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.name, "hms:layers": self.file.layers}


class GageAsset(GenericAsset[GageFile]):
    """Gage asset."""

    regex_parse_str = r".*\.gage$"
    __roles__ = ["hms-gage", MediaType.TEXT]
    __description__ = "Contains data for HEC-HMS gages."
    __file_class__ = GageFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.file.name, HMS_VERSION: self.file.attrs["Version"]} | {
            f"hms:{gage.name}".lower(): dict(gage.attrs.items()) for gage in self.file.gages
        }


class GridAsset(GenericAsset[GridFile]):
    """Grid asset."""

    regex_parse_str = r".*\.grid$"
    __roles__ = ["hms-grid", MediaType.TEXT]
    __description__ = "Contains data for HEC-HMS grid files."
    __file_class__ = GridFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return (
            {HMS_TITLE: self.file.name}
            | {f"hms:{key}".lower(): val for key, val in self.file.attrs.items()}
            | {f"hms:{grid.name}".lower(): dict(grid.attrs.items()) for grid in self.file.grids}
        )


class LogAsset(GenericAsset):
    """Log asset."""

    regex_parse_str = r".*\.log$"
    __roles__ = ["hms-log", "results", MediaType.TEXT]
    __description__ = "Contains log data for HEC-HMS simulations."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.name}


class OutAsset(GenericAsset):
    """Out asset."""

    regex_parse_str = r".*\.out$"
    __roles__ = ["hms-out", "results", MediaType.TEXT]
    __description__ = "Contains output data for HEC-HMS simulations."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.name}


class PdataAsset(GenericAsset[PairedDataFile]):
    """Pdata asset."""

    regex_parse_str = r".*\.pdata$"
    __roles__ = ["hms-pdata", MediaType.TEXT]
    __description__ = "Contains paired data for HEC-HMS simulations."
    __file_class__ = PairedDataFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.file.name, HMS_VERSION: self.file.attrs["Version"]}


class TerrainAsset(GenericAsset[TerrainFile]):
    """Terrain asset."""

    regex_parse_str = r".*\.terrain$"
    __roles__ = ["hms-terrain", MediaType.GEOTIFF]
    __description__ = "Contains terrain data for HEC-HMS simulations."
    __file_class__ = TerrainFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        """Return extra fields with added dynamic keys/values."""
        return {HMS_TITLE: self.file.name, HMS_VERSION: self.file.attrs["Version"]} | {
            f"hms:{layer['name']}".lower(): dict(layer.items()) for layer in self.file.layers
        }


HMS_ASSET_CLASSES = [
    ProjectAsset,
    EventBasinAsset,
    ModelBasinAsset,
    ControlAsset,
    MetAsset,
    SqliteAsset,
    GageAsset,
    RunAsset,
    GridAsset,
    LogAsset,
    OutAsset,
    PdataAsset,
    TerrainAsset,
    DSSAsset,
    GeojsonAsset,
    TiffAsset,
    TiffAsset,
    ThumbnailAsset,
]

HMS_EXTENSION_MAPPING = {re.compile(cls.regex_parse_str, re.IGNORECASE): cls for cls in HMS_ASSET_CLASSES}
