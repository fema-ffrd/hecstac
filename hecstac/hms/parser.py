"""HEC-HMS file parsing classes."""

from __future__ import annotations

import logging
import math
import os
from abc import ABC
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point

import hecstac.hms.utils as utils
from hecstac.hms.consts import BC_LENGTH, BC_LINE_BUFFER, GPD_WRITE_ENGINE
from hecstac.hms.data_model import (
    ET,
    BasinHeader,
    BasinLayerProperties,
    BasinSchematicProperties,
    BasinSpatialProperties,
    ComputationPoints,
    Diversion,
    ElementSet,
    Gage,
    Grid,
    Junction,
    Pattern,
    Precipitation,
    Reach,
    Reservoir,
    Run,
    Sink,
    Source,
    Subbasin,
    Subbasin_ET,
    Table,
    Temperature,
)

logger = logging.getLogger(__name__)


class BaseTextFile(ABC):
    """Base class for text files."""

    def __init__(self, path: str, client=None, bucket=None):
        self.path: str = path
        self.directory: str = os.path.dirname(self.path)
        self.stem: str = os.path.splitext(self.path)[0]
        self.content: str = None
        self.attrs: dict = {}
        self.client = client
        self.bucket = bucket
        self.read_content()
        self.parse_header()

    def read_content(self):
        """Read contents of text file."""
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.content = f.read()
            except UnicodeDecodeError:
                with open(self.path, encoding="cp1252") as f:
                    self.content = f.read()
        else:
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=self.path)
                self.content = response["Body"].read().decode()
            except Exception as e:
                logger.error(e)
                raise FileNotFoundError(f"could not find {self.path} locally nor on s3")

    def parse_header(self):
        """Scan the file down to the first instance of 'End:' and save each colon-separated keyval pair as attrs dict."""
        lines = self.content.splitlines()
        if not lines[0].startswith(
            (
                "Project:",
                "Basin:",
                "Meteorology:",
                "Control:",
                "Terrain Data Manager:",
                "Run:",
                "Paired Data Manager:",
                "Grid Manager",
                "Gage Manager",
            )
        ):
            raise ValueError(f"Unexpected first line: {lines[0]}")
        self.attrs = utils.parse_attrs(lines[1:])


class ProjectFile(BaseTextFile):
    """Class for parsing HEC-HMS project files."""

    def __init__(
        self,
        path: str,
        recurse: bool = True,
        assert_uniform_version: bool = True,
        client=None,
        bucket=None,
    ):
        if not path.endswith(".hms"):
            raise ValueError(f"invalid extension for Project file: {path}")
        super().__init__(path, client=client, bucket=bucket)

        self.basins = []
        self.mets = []
        self.controls = []
        self.terrain = None
        self.run = None
        self.gage = None
        self.grid = None
        self.pdata = None

        if recurse:
            self.scan_for_basins_mets_controls()
            self.scan_for_terrain_run_grid_gage_pdata()
            if assert_uniform_version:
                self.assert_uniform_version()

    def __repr__(self):
        """Representation of the HMSProjectFile class."""
        return f"HMSProjectFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Extract name from project file."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Project: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Project: ") :]

    def combine_stem_ext(self, ext: str) -> str:
        """Combine stem and extension."""
        return f"{self.stem}.{ext}"

    def scan_for_terrain_run_grid_gage_pdata(self):
        """Scan for terrain, run, grid, gage, and pdata files."""
        for ext in ["terrain", "run", "grid", "gage", "pdata"]:
            path = self.combine_stem_ext(ext)
            if os.path.exists(path):
                if ext == "terrain":
                    self.terrain = TerrainFile(path)
                elif ext == "run":
                    self.run = RunFile(path)
                elif ext == "grid":
                    self.grid = GridFile(path)
                elif ext == "gage":
                    self.gage = GageFile(path)
                elif ext == "pdata":
                    self.pdata = PairedDataFile(path)

    def scan_for_basins_mets_controls(self):
        """Scan for basin, meteorology, and control files."""
        lines = self.content.splitlines()
        i = -1
        while True:
            i += 1
            if i >= len(lines):
                break
            line = lines[i]

            if line.startswith("Basin: "):
                nextline = lines[i + 1]
                if not nextline.startswith("     Filename: "):
                    raise ValueError(f"unexpected line: {nextline}")
                basinfile_bn = nextline[len("     Filename: ") :]
                basinfile_path = os.path.join(self.directory, basinfile_bn)
                self.basins.append(BasinFile(basinfile_path))

            if line.startswith("Precipitation: "):
                nextline = lines[i + 1]
                if not nextline.startswith("     Filename: "):
                    raise ValueError(f"unexpected line: {nextline}")
                metfile_bn = nextline[len("     Filename: ") :]
                metfile_path = os.path.join(self.directory, metfile_bn)
                self.mets.append(MetFile(metfile_path))

            if line.startswith("Control: "):
                nextline = lines[i + 1]
                if not nextline.startswith("     FileName: "):
                    raise ValueError(f"unexpected line: {nextline}")
                controlfile_bn = nextline[len("     FileName: ") :]
                controlfile_path = os.path.join(self.directory, controlfile_bn)
                self.controls.append(ControlFile(controlfile_path))

    @property
    def file_counts(self):
        """Return file counts."""
        return {
            "Basins": len(self.basins),
            "Controls": len(self.controls),
            "Mets": len(self.mets),
            "Runs": 1 if self.run is not None else None,
            "Terrain": 1 if self.terrain is not None else None,
            "Paired_Data": 1 if self.pdata is not None else None,
            "Grid": 1 if self.grid is not None else None,
            "Gage": 1 if self.gage is not None else None,
            "SQLite": len([basin.sqlite_path for basin in self.basins if os.path.exists(basin.sqlite_path)]),
        }

    def assert_uniform_version(self):
        """Assert uniform version."""
        errors = []
        version = self.attrs["Version"]
        for basin in self.basins:
            if basin.attrs["Version"] != version:
                errors.append(f"Basin {basin.name} version mismatch (expected {version}, got {basin.attrs['Version']})")
        for met in self.mets:
            if met.attrs["Version"] != version:
                errors.append(
                    f"Meteorology {met.name} version mismatch (expected {version}, got {met.attrs['Version']})"
                )
        for control in self.controls:
            if control.attrs["Version"] != version:
                errors.append(
                    f"Control {control.name} version mismatch (expected {version}, got {control.attrs['Version']})"
                )
        if self.terrain and self.terrain.attrs["Version"] != version:
            errors.append(f"Terrain version mismatch (expected {version}, got {self.terrain.attrs['Version']})")
        # RunFile has no version
        if errors:
            raise ValueError("\n".join(errors))

    @property
    def files(self):
        """Return associated files."""
        # logging.info(f"other paths {[i.path for i in [self.terrain, self.run, self.grid, self.gage, self.pdata] if i]}")
        return (
            [self.path]
            + [basin.path for basin in self.basins]
            + [basin.sqlite_path for basin in self.basins]
            + [control.path for control in self.controls]
            + [met.path for met in self.mets]
            + [i.path for i in [self.terrain, self.run, self.grid, self.gage, self.pdata] if i]
            + self.result_files
            + self.dss_files
        )

    @property
    def dss_files(self):
        """Return dss files."""
        files = set()
        if self.gage:
            files.update(
                [gage.attrs["Variant"]["Variant-1"]["DSS File Name"] for gage in self.gage.elements.elements.values()]
            )
        else:
            logger.warning("No gage file to extract gages from.")

        if self.pdata:
            files.update([pdata.attrs["DSS File"] for pdata in self.pdata.elements.elements.values()])
        else:
            logger.warning("No pdata files found.")

        if self.grid:
            files.update(
                [
                    grid.attrs["Variant"]["Variant-1"]["DSS File Name"]
                    for grid in self.grid.elements.elements.values()
                    if "Variant" in grid.attrs
                ]
            )
        else:
            logger.warning("No grid file to extract dss files from.")

        files = [str(Path(f.replace("\\", "/"))) for f in files]
        return self.absolute_paths(files)

    @property
    def result_files(self):
        """Return result files."""
        if self.run:
            files = set(
                [i[1].attrs["Log File"] for i in self.run.elements]
                + [i[1].attrs["DSS File"] for i in self.run.elements]
                + [i[1].attrs["DSS File"].replace(".dss", ".out") for i in self.run.elements]
            )

            files = [str(Path(f.replace("\\", "/"))) for f in files]
            return self.absolute_paths(set(files))
        else:
            return []

    def absolute_paths(self, paths):
        """Return absolute path."""
        return [os.path.join(self.directory, path) for path in paths]

    @property
    def rasters(self):
        """Return raster files."""
        files = []

        if self.terrain:
            for terrain in self.terrain.layers:
                raster_dir = terrain.get("raster_dir", "").strip()
                if raster_dir and os.path.exists(raster_dir):
                    files += [os.path.join(raster_dir, f) for f in os.listdir(raster_dir)]
                else:
                    logger.warning(f"Skipping missing raster directory: {raster_dir}")

        if self.grid is None:
            logger.warning("No grid file, skipping grid rasters.")
        else:
            files += [
                grid.attrs["Filename"] for grid in self.grid.elements.elements.values() if "Filename" in grid.attrs
            ]
        files = [str(Path(f.replace("\\", "/"))) for f in files]
        return self.absolute_paths(set(files))

    @property
    @lru_cache
    def sqlitedbs(self):
        """Return SQLite database."""
        return [SqliteDB(basin.sqlite_path) for basin in self.basins]


class BasinFile(BaseTextFile):
    """Class for parsing HEC-HMS basin files."""

    def __init__(
        self,
        path: str,
        skip_scans: bool = False,
        client=None,
        bucket=None,
        fiona_aws_session=None,
        read_geom: bool = True,
    ):
        if not path.endswith(".basin"):
            raise ValueError(f"invalid extension for Basin file: {path}")
        super().__init__(path, client=client, bucket=bucket)

        self.header: BasinHeader = ""
        self.layer_properties: BasinLayerProperties = None
        self.spatial_properties: BasinSpatialProperties = None
        self.schematic_properties: BasinSchematicProperties = None
        self.computation_points: ComputationPoints = None
        self.fiona_aws_session = fiona_aws_session
        self.read_geom = read_geom
        self.name = None
        self.parse_name()

        if not skip_scans:
            self.scan_for_headers_and_footers()

        if self.read_geom:
            sqlite_basename = self.identify_sqlite()
            self.sqlite_path = os.path.join(os.path.dirname(self.path), sqlite_basename)

    def __repr__(self):
        """Representation of the HMSBasinFile class."""
        return f"HMSBasinFile({self.path})"

    @property
    def wkt(self):
        """Return wkt representation of the CRS."""
        for line in self.spatial_properties.content.splitlines():
            if "Coordinate System: " in line:
                return line.split(": ")[1]

    @property
    def crs(self):
        """Return the CRS."""
        return CRS(self.wkt)

    @property
    def epsg(self):
        """Return the EPSG code."""
        return self.crs.to_epsg()

    def parse_name(self):
        """Parse basin name."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Basin: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        self.name = lines[0][len("Basin: ") :]

    def scan_for_headers_and_footers(self):
        """Scan for basin headers and footers."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Basin: "):
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.header = BasinHeader(attrs)
            if line.startswith("Basin Schematic Properties:"):
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.schematic_properties = BasinSchematicProperties(attrs)
            if line.startswith("Basin Spatial Properties:"):
                content = "\n".join(utils.get_lines_until_end_sentinel(lines[i + 1 :]))
                self.spatial_properties = BasinSpatialProperties(content)
            if line.startswith("Basin Layer Properties:"):
                content = "\n".join(utils.get_lines_until_end_sentinel(lines[i + 1 :]))
                self.layer_properties = BasinLayerProperties(content)
            if line.startswith("Computation Point:"):
                content = "\n".join(utils.get_lines_until_end_sentinel(lines[i + 1 :]))
                self.computation_points = ComputationPoints(content)

    def identify_sqlite(self):
        """Identify SQLite."""
        for line in self.content.splitlines():
            if ".sqlite" in line:
                return line.split("File: ")[1]

    @property
    @lru_cache
    def elements(self):
        """Return basin elements."""
        elements = ElementSet()
        if self.read_geom:
            sqlite = SqliteDB(
                self.sqlite_path,
                client=self.client,
                bucket=self.bucket,
                fiona_aws_session=self.fiona_aws_session,
            )

            # self.crs = sqlite.crs.to_wkt()
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            geom = None
            slope = None

            if line.startswith("Subbasin: "):
                name = line[len("Subbasin: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                if self.read_geom:
                    geom = sqlite.subbasin_feats[sqlite.subbasin_feats["name"] == name].geometry.values[0]
                elements[name] = Subbasin(name, attrs, geom)

            if line.startswith("Reach: "):
                name = line[len("Reach: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                if self.read_geom:
                    try:
                        geom = sqlite.reach_feats[sqlite.reach_feats["name"] == name].geometry.values[0]
                        if "slope" in sqlite.reach_feats.columns:
                            slope = sqlite.reach_feats[sqlite.reach_feats["name"] == name]["slope"].values[0]
                        else:
                            slope = 0
                    except IndexError:
                        x1 = utils.search_contents(lines[i + 1 :], "Canvas X", ":", False)[0]
                        y1 = utils.search_contents(lines[i + 1 :], "Canvas Y", ":", False)[0]
                        x2 = utils.search_contents(lines[i + 1 :], "From Canvas X", ":", False)[0]
                        y2 = utils.search_contents(lines[i + 1 :], "From Canvas Y", ":", False)[0]
                        geom = LineString([[float(x1), float(y1)], [float(x2), float(y2)]])
                        slope = 0

                elements[name] = Reach(name, attrs, geom, slope)

            if line.startswith("Junction: "):
                name = line[len("Junction: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                if self.read_geom:
                    geom = Point((float(attrs["Canvas X"]), float(attrs["Canvas Y"])))
                elements[name] = Junction(name, attrs, geom)

            if line.startswith("Sink: "):
                name = line[len("Sink: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                if self.read_geom:
                    geom = Point((float(attrs["Canvas X"]), float(attrs["Canvas Y"])))
                elements[name] = Sink(name, attrs, geom)

            if line.startswith("Reservoir: "):
                name = line[len("Reservoir: ") :]
                attrs = OrderedDict({"text": lines[i + 1 :]})
                if self.read_geom:
                    x = utils.search_contents(lines[i + 1 :], "Canvas X", ":", False)[0]
                    y = utils.search_contents(lines[i + 1 :], "Canvas Y", ":", False)[0]
                    geom = Point((float(x), float(y)))
                elements[name] = Reservoir(name, attrs, geom)

            if line.startswith("Source: "):
                name = line[len("Source: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                if self.read_geom:
                    geom = Point((float(attrs["Canvas X"]), float(attrs["Canvas Y"])))
                elements[name] = Source(name, attrs, geom)

            if line.startswith("Diversion: "):
                name = line[len("Diversion: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                if self.read_geom:
                    geom = Point((float(attrs["Canvas X"]), float(attrs["Canvas Y"])))
                elements[name] = Diversion(name, attrs, geom)

        return elements

    @property
    @lru_cache
    def subbasins(self):
        """Return subbasin elements."""
        return self.elements.get_element_type("Subbasin")

    @property
    @lru_cache
    def reaches(self):
        """Return reach elements."""
        return self.elements.get_element_type("Reach")

    @property
    @lru_cache
    def junctions(self):
        """Return junction elements."""
        return self.elements.get_element_type("Junction")

    @property
    @lru_cache
    def reservoirs(self):
        """Return reservoir elements."""
        return self.elements.get_element_type("Reservoir")

    @property
    @lru_cache
    def diversions(self):
        """Return diversion elements."""
        return self.elements.get_element_type("Diversion")

    @property
    @lru_cache
    def sinks(self):
        """Return sink elements."""
        return self.elements.get_element_type("Sink")

    @property
    @lru_cache
    def sources(self):
        """Return source elements."""
        return self.elements.get_element_type("Source")

    @property
    @lru_cache
    def gages(self):
        """Return gages."""
        return self.elements.gages

    @property
    @lru_cache
    def drainage_area(self):
        """Return drainage areas.."""
        return sum([subbasin.geom.area for subbasin in self.subbasins])

    @property
    @lru_cache
    def reach_miles(self):
        """Return reach lengths in miles.."""
        return sum([reach.geom.length for reach in self.reaches])

    @property
    @lru_cache
    def basin_geom(self):
        """Return basin geometry."""
        return utils.remove_holes(self.feature_2_gdf("Subbasin").make_valid().to_crs(4326).union_all())

    def bbox(self, crs):
        """Return basin bounding box."""
        return self.feature_2_gdf("Subbasin").to_crs(crs).total_bounds

    def feature_2_gdf(self, element_type: str) -> gpd.GeoDataFrame:
        """Convert feature to GeoDataFrame."""
        gdf_list = []
        for e in self.elements.get_element_type(element_type):
            gdf_list.append(
                gpd.GeoDataFrame([{"name": e.name, "geometry": e.geom} | e.attrs], geometry="geometry", crs=self.crs)
            )
        if len(gdf_list) == 1:
            return gdf_list[0]
        elif len(gdf_list) == 0:
            return None
        else:
            return pd.concat(gdf_list)

    @property
    @lru_cache
    def observation_points_gdf(self):
        """Return GeoDataFrame of observation points."""
        gdf_list = []
        for name, element in self.elements:
            if "Observed Hydrograph Gage" in element.attrs.keys():
                if isinstance(element, Junction) or isinstance(element, Sink):
                    gdf_list.append(
                        gpd.GeoDataFrame(
                            {
                                "name": name,
                                "geometry": element.geom,
                                "gage_name": element.attrs["Observed Hydrograph Gage"],
                            },
                            geometry="geometry",
                            crs=self.crs,
                            index=[0],
                        )
                    )
                elif isinstance(element, Subbasin):
                    gdf_list.append(
                        gpd.GeoDataFrame(
                            {
                                "name": name,
                                "geometry": element.geom.centroid,
                                "gage_name": element.attrs["Observed Hydrograph Gage"],
                            },
                            geometry="geometry",
                            crs=self.crs,
                            index=[0],
                        )
                    )
                elif isinstance(element, Reach):
                    start_point = element.geom.boundary
                    gdf_list.append(
                        gpd.GeoDataFrame(
                            {
                                "name": name,
                                "geometry": start_point,
                                "gage_name": element.attrs["Observed Hydrograph Gage"],
                            },
                            geometry="geometry",
                            crs=self.crs,
                            index=[0],
                        )
                    )
        if len(gdf_list) == 1:
            return gdf_list[0]
        elif len(gdf_list) == 0:
            return None
        else:
            gdf = gpd.GeoDataFrame(pd.concat(gdf_list), crs=self.crs, geometry="geometry")
            return gdf

    def subbasin_connection_lines(self) -> gpd.GeoDataFrame:
        """Return GeoDataframe of subbasin connection lines."""
        df_list = []
        for subbasin in self.subbasins:
            us_point = subbasin.geom.centroid
            ds_element = self.elements[subbasin.attrs["Downstream"]]
            if ds_element in self.reaches:
                ds_point = Point(ds_element.geom.coords[-1])
            else:
                ds_point = ds_element.geom
            df = pd.DataFrame(subbasin.attrs, index=[0])
            if not us_point.equals(ds_point):
                df["us_name"], df["ds_name"], df["geometry"] = (
                    subbasin.name,
                    ds_element.name,
                    LineString([us_point, ds_point]),
                )
            df_list.append(df)
        df = pd.concat(df_list)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=self.crs)
        return gdf

    def junction_connection_lines(self) -> gpd.GeoDataFrame:
        """Return GeoDataframe of junction connection lines."""
        df_list = []
        for junction in self.junctions:
            us_point = junction.geom
            if "Downstream" not in junction.attrs:
                logger.warning(f"Warning no downstream element for junction {junction.name}")
                continue
            ds_element = self.elements[junction.attrs["Downstream"]]
            if ds_element in self.reaches:
                if isinstance(ds_element.geom, LineString):
                    ds_point = Point(ds_element.geom.coords[-1])
                elif isinstance(ds_element.geom, MultiLineString):
                    ds_point = Point(ds_element.geom.geoms[0].coords[-1])
                else:
                    raise TypeError(
                        f"Expected either LineString or MultiLineString for reaches; recieved {type(ds_element.geom)}"
                    )
            else:
                ds_point = ds_element.geom
            df = pd.DataFrame(junction.attrs, index=[0])
            df["us_point"] = us_point
            if not us_point.equals(ds_point):
                df["us_name"], df["ds_name"], df["geometry"] = (
                    junction.name,
                    ds_element.name,
                    LineString([Point(us_point.x, us_point.y), Point(ds_point.x, ds_point.y)]),
                )
            df_list.append(df)
        df = pd.concat(df_list)
        if "geometry" in df.columns:
            df = df.drop(columns=["us_point"])
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=self.crs)
        else:
            gdf = gpd.GeoDataFrame(df, geometry="us_point", crs=self.crs)
        return gdf

    @property
    @lru_cache
    def hms_schematic_2_gdfs(self) -> dict[gpd.GeoDataFrame]:
        """Convert HMS schematics to GeoDataframe."""
        element_gdfs = {}
        for element_type in [
            "Reach",
            "Subbasin",
            "Junction",
            "Diversion",
            "Source",
            "Sink",
            "Reservoir",
        ]:
            if self.elements.get_element_type(element_type):
                element_gdfs[element_type] = self.feature_2_gdf(element_type)
        element_gdfs["Subbasin_Connectors"] = self.subbasin_connection_lines()
        element_gdfs["Junction_Connectors"] = self.junction_connection_lines()
        element_gdfs["Recommended_BC_Lines"] = self.subbasin_bc_lines()
        return element_gdfs

    def subbasin_bc_lines(self):
        """Return subbasin boundary condition lines."""
        df_list = []
        for _, row in self.subbasin_connection_lines().iterrows():
            geom = row.geometry
            p1 = Point(geom.coords[0])
            p2 = Point(geom.coords[1])
            p3 = geom.interpolate(geom.length - BC_LINE_BUFFER)
            reach_angle = math.atan2(p2.y - p1.y, p2.x - p1.x)  # atan2(y1 - y0, x1 - x0)
            # rotate cross section direction to be quarter of a turn clockwise from reach direction
            bc_angle = reach_angle - math.pi / 2
            x_start = p3.x - math.cos(bc_angle) * BC_LENGTH / 2
            y_start = p3.y - math.sin(bc_angle) * BC_LENGTH / 2
            x_end = p3.x + math.cos(bc_angle) * BC_LENGTH / 2
            y_end = p3.y + math.sin(bc_angle) * BC_LENGTH / 2
            bc_geom = LineString(((x_start, y_start), (x_end, y_end)))
            df_list.append(pd.DataFrame([[row["us_name"], bc_geom]], columns=["name", "geometry"]))
        df = pd.concat(df_list)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=self.crs)
        return gdf

    @property
    @lru_cache
    def hms_methods(self):
        """Extract unique HMS methods from Subbasins and Reaches."""
        methods = {
            "Canopy": set(),
            "Discretization": set(),
            "Begin Snow": set(),
            "Surface": set(),
            "LossRate": set(),
            "Transform": set(),
            "Baseflow": set(),
            "Route": set(),
        }

        for subbasin in self.subbasins:
            for key in methods.keys():
                if key in subbasin.attrs:
                    methods[key].add(subbasin.attrs[key])
        for reach in self.reaches:
            if "Route" in reach.attrs:
                methods["Route"].add(reach.attrs["Route"])
        return {f"hms_methods:{key.replace(' ', '_')}": list(values) for key, values in methods.items()}


class MetFile(BaseTextFile):
    """Class for parsing HEC-HMS meteorology files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".met"):
            raise ValueError(f"invalid extension for Meteorology file: {path}")
        super().__init__(path, client=client, bucket=bucket)

        self.scan_for_elements()

    def __repr__(self):
        """Representation of the HMSMetFile class."""
        return f"HMSMetFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Return meteorology name."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Meteorology: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Meteorology: ") :]

    def scan_for_elements(self):
        """Scan for meteorology elements."""
        elements = ElementSet()
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Precip Method Parameters: "):
                name = line[len("Precip Method Parameters: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                elements[name] = Precipitation(name=name, attrs=attrs)

            elif line.startswith("Air Temperature Method Parameters: "):
                name = line[len("Air Temperature Method Parameters: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                elements[name] = Temperature(name=name, attrs=attrs)

            elif line.startswith("Evapotranspiration Method Parameters: "):
                name = line[len("Evapotranspiration Method Parameters: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                elements[name] = ET(name=name, attrs=attrs)

            elif line.startswith("Subbasin: "):
                name = line[len("Subbasin: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                elements[name] = Subbasin_ET(name=name, attrs=attrs)
        self.elements = elements


class ControlFile(BaseTextFile):
    """Class for parsing HEC-HMS control files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".control"):
            raise ValueError(f"invalid extension for Control file: {path}")
        super().__init__(path, client=client, bucket=bucket)

    def __repr__(self):
        """Representation of the HMSControlFile class."""
        return f"HMSControlFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Return control name."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Control: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Control: ") :]


class TerrainFile(BaseTextFile):
    """Class for parsing HEC-HMS terrain files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".terrain"):
            raise ValueError(f"Invalid extension for Terrain file: {path}")
        super().__init__(path, client=client, bucket=bucket)
        self.layers = []

        found_first = False
        name, raster_path, raster_dir, vert_units = "", "", "", ""

        for line in self.content.splitlines():
            if not found_first:
                if line.startswith("Terrain Data: "):
                    found_first = True
                else:
                    continue

            if line == "End:":
                self.layers.append(
                    {
                        "name": name,
                        "raster_path": raster_path,
                        "raster_dir": raster_dir,
                        "vert_units": vert_units,
                    }
                )
                name, raster_path, raster_dir, vert_units = "", "", "", ""

            elif line.startswith("Terrain Data: "):
                name = line[len("Terrain Data: ") :]
            elif line.startswith("     Elevation File Name: "):
                raster_path_raw = line[len("     Elevation File Name: ") :]
                raster_path = os.path.join(os.path.dirname(self.path), raster_path_raw.replace("\\", os.sep))
            elif line.startswith("     Terrain Directory: "):
                raster_dir_raw = line[len("     Terrain Directory: ") :]
                raster_dir = os.path.join(os.path.dirname(self.path), raster_dir_raw.replace("\\", os.sep))
            elif line.startswith("     Vertical Units: "):
                vert_units = line[len("     Vertical Units: ") :]

    def __repr__(self):
        """Representation of the HMSTerrainFile class."""
        return f"HMSTerrainFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Return name."""
        return None


class RunFile(BaseTextFile):
    """Class for parsing HEC-HMS run files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".run"):
            raise ValueError(f"invalid extension for Run file: {path}")
        super().__init__(path, client=client, bucket=bucket)

    def __repr__(self):
        """Representation of the HMSRunFile class."""
        return f"HMSRunFile({self.path})"

    def runs(self):
        """Retrieve all runs."""
        runs = ElementSet()
        lines = self.content.splitlines()
        i = -1
        while True:
            i += 1
            if i >= len(lines):
                break
            line = lines[i]
            if line.startswith("Run: "):
                name = line.split("Run: ")[1]
                runs[name] = Run(name, utils.parse_attrs(lines[i + 1 :]))
        return runs

    @property
    def elements(self):
        """Return run elements."""
        return self.runs()


class PairedDataFile(BaseTextFile):
    """Class for parsing HEC-HMS paired data files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".pdata"):
            raise ValueError(f"invalid extension for Paired Data file: {path}")
        if not os.path.exists(path):
            try:
                response = client.get_object(Bucket=bucket, Key=path)
                self.content = response["Body"].read().decode()
            except Exception as e:
                logger.info(f" {e}: No Paired Data File found: creating empty Paired Data File")
                self.create_pdata(path)
        super().__init__(path, client=client, bucket=bucket)
        self.elements = ElementSet()
        self.scan_for_tables()

    def __repr__(self):
        """Representation of the HMSPairedDataFile class."""
        return f"HMSPairedDataFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Return paired data manager."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Paired Data Manager: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Paired Data Manager: ") :]

    def scan_for_tables(self):
        """Scan for tables."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Table: "):
                name = line[len("Table: ") :]
                table_type = lines[i + 1][len("     Table Type: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[f"{name}+{table_type}"] = Table(name, attrs)

    def scan_for_patterns(self):
        """Scan for patterns."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Pattern: "):
                name = line[len("Pattern: ") :]
                data_type = lines[i + 1][len("     Data Type: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[f"{name}+{data_type}"] = Pattern(name, attrs)


class SqliteDB:
    """SQLite database class."""

    def __init__(self, path: str, client=None, bucket=None, fiona_aws_session=None):
        sqlite_file, _ = os.path.splitext(path)
        path = f"{sqlite_file}.sqlite"
        if not path.endswith(".sqlite"):
            raise ValueError(f"invalid extension for sqlite database: {path}")
        self.path = path
        self.client = client
        self.bucket = bucket
        self.fiona_aws_session = fiona_aws_session
        if not os.path.exists(self.path):
            self.path = f"s3://{self.bucket}/{self.path}"
        if self.fiona_aws_session:
            with fiona.Env(self.fiona_aws_session):
                self.layers = fiona.listlayers(self.path)
                self.reach_feats = gpd.read_file(self.path, layer="reach2d", engine=GPD_WRITE_ENGINE)
                self.subbasin_feats = gpd.read_file(self.path, layer="subbasin2d", engine=GPD_WRITE_ENGINE)
        else:
            self.layers = fiona.listlayers(self.path)
            self.reach_feats = gpd.read_file(self.path, layer="reach2d", engine=GPD_WRITE_ENGINE)
            self.subbasin_feats = gpd.read_file(self.path, layer="subbasin2d", engine=GPD_WRITE_ENGINE)

        # check consistent crs and assign crs to sqlite class
        if (
            self.reach_feats.crs != self.subbasin_feats.crs
        ):  # could also compare to coordinate system in the .basin file. once we parse that
            raise ValueError("coordinate system misalignment between subbasins and reaches")
        # else:
        #     self.crs = self.subbasin_feats.crs


class GridFile(BaseTextFile):
    """Class for parsing HEC-HMS grid files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".grid"):
            raise ValueError(f"invalid extension for Grid file: {path}")
        super().__init__(path, client=client, bucket=bucket)
        self.elements = ElementSet()
        self.scan_for_grids()

    def __repr__(self):
        """Representation of the HMSGridFile class."""
        return f"HMSGridFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Return grid manager name."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Grid Manager: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Grid Manager: ") :]

    def scan_for_grids(self):
        """Scan for all grids."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Grid: "):
                name = line[len("Grid: ") :]
                grid_type = lines[i + 1][len("     Grid Type: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[f"{name}+{grid_type}"] = Grid(f"{name}+{grid_type}", attrs)

    def remove_grid_type(self, grid_types: list[str]):
        """Remove given grid types."""
        new_elements = ElementSet()
        for name, g in self.elements.elements.items():
            if g.attrs["Grid Type"] not in grid_types:
                new_elements[name] = g
        self.elements = new_elements

    @property
    @lru_cache
    def grids(self):
        """Return grid elements."""
        return self.elements.get_element_type("Grid")


class GageFile(BaseTextFile):
    """Class for parsing HEC-HMS gage files."""

    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".gage"):
            raise ValueError(f"invalid extension for Gage file: {path}")
        super().__init__(path, client=client, bucket=bucket)
        self.elements = ElementSet()
        self.scan_for_gages()

    def __repr__(self):
        """Representation of the HMSGageFile class."""
        return f"HMSGageFile({self.path})"

    @property
    @lru_cache
    def name(self):
        """Return gage manager name."""
        lines = self.content.splitlines()
        if not lines[0].startswith("Gage Manager: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Gage Manager: ") :]

    def scan_for_gages(self):
        """Search for all gages."""
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Gage: "):
                name = line[len("Gage: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[name] = Gage(name, attrs)

    @property
    @lru_cache
    def gages(self):
        """Return gage elements."""
        return self.elements.get_element_type("Gage")
