"""HEC-HMS Stac Item asset classes."""

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


class GeojsonAsset(GenericAsset):
    """Geojson asset."""

    __roles__ = ["data", MediaType.GEOJSON]
    __description__ = "Geojson file."


class TiffAsset(GenericAsset):
    """Tiff Asset."""

    __roles__ = ["data", MediaType.GEOTIFF]
    __description__ = "Tiff file."


class ProjectAsset(GenericAsset[ProjectFile]):
    """HEC-HMS Project file asset."""

    __roles__ = ["hms-project", MediaType.TEXT]
    __description__ = "The HEC-HMS project file. Summary provied at the item level"
    __file_class__ = ProjectFile


class ThumbnailAsset(GenericAsset):
    """Thumbnail asset."""

    __roles__ = ["thumbnail", MediaType.PNG]
    __description__ = "Thumbnail"


class ModelBasinAsset(GenericAsset[BasinFile]):
    """HEC-HMS Basin file asset from authoritative model, containing geometry and other detailed data."""

    __roles__ = ["hms-basin", MediaType.TEXT]
    __description__ = "Defines the basin geometry and elements for HEC-HMS simulations."
    __file_class__ = BasinFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {
            "hms:title": self.file.name,
            "hms:version": self.file.header.attrs["Version"],
            "hms:description": self.file.header.attrs.get("Description"),
            "hms:unit_system": self.file.header.attrs["Unit System"],
            "hms:gages": self.file.gages,
            "hms:drainage_area_miles": self.file.drainage_area,
            "hms:reach_length_miles": self.file.reach_miles,
            "proj:wkt": self.file.wkt,
            "proj:code": self.file.epsg,
        } | {f"hms_basin:{key}".lower(): val for key, val in self.file.elements.element_counts.items()}


class EventBasinAsset(GenericAsset[BasinFile]):
    """HEC-HMS Basin file asset from event, with limited basin info."""

    __roles__ = ["hms-basin", MediaType.TEXT]
    __description__ = "Defines the basin geometry and elements for HEC-HMS simulations."
    __file_class__ = BasinFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {
            "hms:title": self.file.name,
            "hms:version": self.file.header.attrs["Version"],
            "hms:description": self.file.header.attrs.get("Description"),
            "hms:unit_system": self.file.header.attrs["Unit System"],
        }


class RunAsset(GenericAsset[RunFile]):
    """Run asset."""

    __file_class__ = RunFile
    __roles__ = ["hms-run", MediaType.TEXT]
    __description__ = "Contains data for HEC-HMS simulations."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.name} | {
            run.name: {f"hms:{key}".lower(): val for key, val in run.attrs.items()} for _, run in self.file.elements
        }


class ControlAsset(GenericAsset[ControlFile]):
    """HEC-HMS Control file asset."""

    __roles__ = ["hms-control", MediaType.TEXT]
    __description__ = "Defines time control information for HEC-HMS simulations."
    __file_class__ = ControlFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {
            "hms:title": self.file.name,
            **{f"hms:{key}".lower(): val for key, val in self.file.attrs.items()},
        }


class MetAsset(GenericAsset[MetFile]):
    """HEC-HMS Meteorological file asset."""

    __roles__ = ["hms-met", MediaType.TEXT]
    __description__ = "Contains meteorological data such as precipitation and temperature."
    __file_class__ = MetFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {
            "hms:title": self.file.name,
            **{f"hms:{key}".lower(): val for key, val in self.file.attrs.items()},
        }


class DSSAsset(GenericAsset):
    """DSS asset."""

    __roles__ = ["hec-dss", "application/octet-stream"]
    __description__ = "HEC-DSS file."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.name}


class SqliteAsset(GenericAsset[SqliteDB]):
    """HEC-HMS SQLite database asset."""

    __roles__ = ["hms-sqlite", "application/x-sqlite3"]
    __description__ = "Stores spatial data for HEC-HMS basin files."
    __file_class__ = SqliteDB

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.name, "hms:layers": self.file.layers}


class GageAsset(GenericAsset[GageFile]):
    """Gage asset."""

    __roles__ = ["hms-gage", MediaType.TEXT]
    __description__ = "Contains data for HEC-HMS gages."
    __file_class__ = GageFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.file.name, "hms:version": self.file.attrs["Version"]} | {
            f"hms:{gage.name}".lower(): {key: val for key, val in gage.attrs.items()} for gage in self.file.gages
        }


class GridAsset(GenericAsset[GridFile]):
    """Grid asset."""

    __roles__ = ["hms-grid", MediaType.TEXT]
    __description__ = "Contains data for HEC-HMS grid files."
    __file_class__ = GridFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return (
            {"hms:title": self.file.name}
            | {f"hms:{key}".lower(): val for key, val in self.file.attrs.items()}
            | {f"hms:{grid.name}".lower(): {key: val for key, val in grid.attrs.items()} for grid in self.file.grids}
        )


class LogAsset(GenericAsset):
    """Log asset."""

    __roles__ = ["hms-log", "results", MediaType.TEXT]
    __description__ = "Contains log data for HEC-HMS simulations."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.name}


class OutAsset(GenericAsset):
    """Out asset."""

    __roles__ = ["hms-out", "results", MediaType.TEXT]
    __description__ = "Contains output data for HEC-HMS simulations."

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.name}


class PdataAsset(GenericAsset[PairedDataFile]):
    """Pdata asset."""

    __roles__ = ["hms-pdata", MediaType.TEXT]
    __description__ = "Contains paired data for HEC-HMS simulations."
    __file_class__ = PairedDataFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.file.name, "hms:version": self.file.attrs["Version"]}


class TerrainAsset(GenericAsset[TerrainFile]):
    """Terrain asset."""

    __roles__ = ["hms-terrain", MediaType.GEOTIFF]
    __description__ = "Contains terrain data for HEC-HMS simulations."
    __file_class__ = TerrainFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self):
        return {"hms:title": self.file.name, "hms:version": self.file.attrs["Version"]} | {
            f"hms:{layer['name']}".lower(): {key: val for key, val in layer.items()} for layer in self.file.layers
        }


HMS_EXTENSION_MAPPING = {
    ".hms": ProjectAsset,
    ".basin": {"event": EventBasinAsset, "model": ModelBasinAsset},
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
    ".tif": TiffAsset,
    ".png": ThumbnailAsset,
}
