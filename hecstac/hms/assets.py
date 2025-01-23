from pathlib import Path

from hms.parser import (
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
from hms.s3_utils import check_storage_extension
from pystac import Asset, MediaType


def asset_factory(fpath: str) -> Asset:
    """Detemine the type of file and resturn the corresponding asset type."""
    fpath = str(fpath)
    file_extension = Path(fpath).suffix.lower()

    if file_extension == ".hms":
        roles = ["hms-project"]
        description = """The HEC-HMS project file."""
        asset = ProjectAsset(fpath, roles=roles, description=description)
    elif file_extension == ".basin":
        roles = [
            "hms-basin",
        ]
        description = """HEC-HMS basin file. Describes the physical characteristics of the watershed, including sub-basins, reaches, junctions, reservoirs, and other hydrologic elements."""
        asset = BasinAsset(fpath, roles=roles, description=description)
    elif file_extension == ".control":
        roles = ["hms-control"]
        description = """HEC-HMS control file. Defines the time control information for the simulation, including start and end times, time step, and other temporal parameters."""
        asset = ControlAsset(fpath, roles=roles, description=description)
    elif file_extension == ".met":
        roles = ["hms-met"]
        description = """HEC-HMS meteorolgical file. Contains meteorological data such as precipitation, temperature, and evapotranspiration data."""
        asset = MetAsset(fpath, roles=roles, description=description)
    elif file_extension == ".dss":
        roles = ["hms-dss"]
        description = """Hydrologic Engineering Center Data Storage System (HEC-DSS) file. Stores time series data, paired data, and other types of data."""
        asset = DSSAsset(fpath, roles=roles, description=description)
    elif file_extension == ".sqlite":
        roles = ["hms-sqlite"]
        description = (
            """A SQLite database file that stores spatial data corresponding to a basin file of the same name."""
        )
        asset = SqliteAsset(fpath, roles=roles, description=description)
    elif file_extension == ".gage":
        roles = ["hms-gage"]
        description = """HEC-HMS gage file. Contains information about gages used in the model, including location, type, and observed data."""
        asset = GageAsset(fpath, roles=roles, description=description)
    elif file_extension == ".log":
        roles = ["hms-log"]
        description = """HEC-HMS log file. Contains log information from the model run, including errors, warnings, and other messages."""
        asset = LogAsset(fpath, roles=roles, description=description)
    elif file_extension == ".out":
        roles = ["hms-out"]
        description = """HEC-HMS output file. Contains log information from the model run, including errors, warnings, and other messages."""
        asset = OutAsset(fpath, roles=roles, description=description)
    elif file_extension == ".pdata":
        roles = ["hms-paired-data"]
        description = """HEC-HMS paired data file. Stores paired data, such as rating curves or other relationships used in the model."""
        asset = PdataAsset(fpath, roles=roles, description=description)
    elif file_extension == ".terrain":
        roles = ["hms-terrain"]
        description = """HEC-HMS terrain file. Contains elevation data for the watershed, used to define the topography and flow paths."""
        asset = TerrainAsset(fpath, roles=roles, description=description)
    elif file_extension == ".grid":
        roles = ["hms-grid"]
        description = """HEC-HMS grid file. Contains grid info and mapping to HEC-DSS grid paths."""
        asset = GridAsset(fpath, roles=roles, description=description)
    elif file_extension == ".run":
        roles = ["hms-run"]
        description = """HEC-HMS run file. Contains run info and mapping of met, control, basin and output files."""
        asset = RunAsset(fpath, roles=roles, description=description)
    elif file_extension == ".png":
        roles = ["thumbnail"]
        description = """Thumbnail depicting the model geometry."""
        asset = ThumbnailAsset(fpath, roles=roles, description=description)
    elif file_extension == ".tif":
        roles = ["raster"]
        description = """Raster file."""
        asset = TiffAsset(fpath, roles=roles, description=description)
    elif file_extension == ".geojson":
        roles = ["model-geometry"]
        description = """Model geometry."""
        asset = GeojsonAsset(fpath, roles=roles, description=description)
    else:
        asset = Asset(fpath)
        asset.title = Path(fpath).name

    asset.title = Path(fpath).name
    asset = check_storage_extension(asset)
    return asset


class GenericAsset(Asset):
    """Generic Asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.name = Path(href).name
        self.stem = Path(href).stem

    def name_from_suffix(self, suffix: str) -> str:
        return self.stem + "." + suffix


class GeojsonAsset(GenericAsset):
    """Geojson asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.media_type = MediaType.GEOJSON


class TiffAsset(GenericAsset):
    """Tiff Asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.roles.append(MediaType.TIFF)


class ProjectAsset(GenericAsset):
    "Project Asset."

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.pf = ProjectFile(href, assert_uniform_version=False)
        self.extra_fields = {
            "hms:project_title": self.pf.name,
            "hms:version": self.pf.attrs["Version"],
            "hms:description": self.pf.attrs.get("Description"),
            "hms:unit system": self.pf.basins[0].header.attrs["Unit System"],
        } | {f"hms:{key}": val for key, val in self.pf.file_counts.items()}

        self.roles.append(MediaType.TEXT)


class ThumbnailAsset(GenericAsset):
    """Thumbnail asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.roles.append(MediaType.PNG)


class BasinAsset(GenericAsset):
    """Basin asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.bf = BasinFile(href)
        self.extra_fields = {
            "hms:title": self.bf.name,
            "hms:version": self.bf.header.attrs["Version"],
            "hms:description": self.bf.header.attrs.get("Description"),
            "hms:unit system": self.bf.header.attrs["Unit System"],
            "hms:gages": self.bf.gages,
            "hms:drainage area miles": self.bf.drainage_area,
            "hms:reach length miles": self.bf.reach_miles,
            "projection:wkt": self.bf.wkt,
            "projection:code": self.bf.epsg,
        } | {f"hms:{key}".lower(): val for key, val in self.bf.elements.element_counts.items()}

        self.roles.append(MediaType.TEXT)


class RunAsset(GenericAsset):
    """Run asset."""

    def __init__(self, href: str, *args, **kwargs):
        self.rf = RunFile(href)
        super().__init__(href, *args, **kwargs)
        self.extra_fields = {"hms:title": self.name} | {
            run.name: {f"hms:{key}".lower(): val for key, val in run.attrs.items()} for _, run in self.rf.elements
        }

        self.roles.append(MediaType.TEXT)


class ControlAsset(GenericAsset):
    """Control asset."""

    def __init__(self, href: str, *args, **kwargs):
        self.cf = ControlFile(href)
        super().__init__(href, *args, **kwargs)
        self.extra_fields = {"hms:title": self.cf.name} | {
            f"hms:{key}".lower(): val for key, val in self.cf.attrs.items()
        }

        self.roles.append(MediaType.TEXT)


class MetAsset(GenericAsset):
    "Met asset."

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.mf = MetFile(href)
        self.extra_fields = {"hms:title": self.mf.name} | {
            f"hms:{key}".lower(): val for key, val in self.mf.attrs.items()
        }

        self.roles.append(MediaType.TEXT)


class DSSAsset(GenericAsset):
    """DSS asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)

        self.extra_fields["hms:title"] = self.name


class SqliteAsset(GenericAsset):

    def __init__(self, href, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.sqdb = SqliteDB(href)
        self.extra_fields = {"hms:title": self.name, "hsm:layers": self.sqdb.layers}


class GageAsset(GenericAsset):
    """Gage asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.gf = GageFile(href)
        self.extra_fields = {"hms:title": self.gf.name, "hms:version": self.gf.attrs["Version"]} | {
            f"hms:{gage.name}".lower(): {key: val for key, val in gage.attrs.items()} for gage in self.gf.gages
        }

        self.roles.append(MediaType.TEXT)


class GridAsset(GenericAsset):
    """Grid asset"""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.gf = GridFile(href)
        self.extra_fields = (
            {"hms:title": self.gf.name}
            | {f"hms:{key}".lower(): val for key, val in self.gf.attrs.items()}
            | {f"hms:{grid.name}".lower(): {key: val for key, val in grid.attrs.items()} for grid in self.gf.grids}
        )

        self.roles.append(MediaType.TEXT)


class LogAsset(GenericAsset):
    """Log asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.extra_fields["hms:title"] = self.name

        self.roles.append(MediaType.TEXT)


class OutAsset(GenericAsset):
    """Out asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.extra_fields["hms:title"] = self.name

        self.roles.append(MediaType.TEXT)


class PdataAsset(GenericAsset):
    """Pdata asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.pd = PairedDataFile(href)
        self.extra_fields = {"hms:title": self.pd.name, "hms:version": self.pd.attrs["Version"]}

        self.roles.append(MediaType.TEXT)


class TerrainAsset(GenericAsset):
    """Terrain asset."""

    def __init__(self, href: str, *args, **kwargs):
        super().__init__(href, *args, **kwargs)
        self.tf = TerrainFile(href)
        self.extra_fields = {"hms:title": self.tf.name, "hms:version": self.tf.attrs["Version"]} | {
            f"hms:{layer['name']}".lower(): {key: val for key, val in layer.items()} for layer in self.tf.layers
        }

        self.roles.append(MediaType.TEXT)
