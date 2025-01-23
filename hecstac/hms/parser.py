from __future__ import annotations

import math
import os
from abc import ABC
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache

import fiona
import geopandas as gpd
import hms.utils as utils
import pandas as pd
from hms.consts import (
    BASIN_DEFAULTS,
    BC_LENGTH,
    BC_LINE_BUFFER,
    CONTROL_DEFAULTS,
    DEFAULT_BASIN_HEADER,
    DEFAULT_BASIN_LAYER_PROPERTIES,
    GAGE_DEFAULTS,
    GPD_WRITE_ENGINE,
    GRID_DEFAULTS,
    RUN_DEFAULTS,
)
from hms.data_model import (
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
from pyproj import CRS
from shapely import get_point
from shapely.geometry import LineString, MultiLineString, Point


class BaseTextFile(ABC):
    def __init__(self, path: str, client=None, bucket=None, new_file: bool = False):
        self.path: str = path
        self.directory: str = os.path.dirname(self.path)
        self.stem: str = os.path.splitext(self.path)[0]
        self.content: str = None
        self.attrs: dict = {}
        self.client = client
        self.bucket = bucket

        if not new_file:
            self.read_content()
            self.parse_header()

    def read_content(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.content = f.read()
        else:
            try:
                response = self.client.get_object(Bucket=self.bucket, Key=self.path)
                self.content = response["Body"].read().decode()
            except Exception as E:
                print(E)
                raise FileNotFoundError(f"could not find {self.path} locally nor on s3")

    def parse_header(self):
        """Scan the file down to the first instance of 'End:' and save each colon-separated keyval pair as attrs dict"""
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

    def write(self, path=None, s3=False, new_file=False):
        if not path:
            path = self.path
        print(f"writing: {path}")
        if not new_file:
            if os.path.exists(path):
                with open(path, "w") as f:
                    f.write(self.content)
            else:
                self.client.put_object(Body=self.content, Bucket=self.bucket, Key=path)
        else:
            with open(path, "w") as f:
                f.write(self.content)


class ProjectFile(BaseTextFile):
    def __init__(
        self,
        path: str,
        recurse: bool = True,
        assert_uniform_version: bool = True,
        client=None,
        bucket=None,
        new_file=False,
    ):
        if not path.endswith(".hms"):
            raise ValueError(f"invalid extension for Project file: {path}")
        super().__init__(path, client=client, bucket=bucket, new_file=new_file)

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

    def new_project(
        self,
        title: str,
        description: str = "",
        version: str = "4.11",
        dss_file_name: str = None,
        time_zone: str = "America/Chicago",
    ):
        """Create a new HEC-HMS project."""
        lines = [f"Project: {title}"]
        lines.append(f"     Description: {description}")
        lines.append(f"     Version: {version}")
        lines.append("     Filepath Seperator: \\")
        if dss_file_name is None:
            lines.append(f"     Dss File Name: {title}.dss")
        else:
            lines.append(f"     Dss File Name: {dss_file_name}.dss")
        lines.append(f"     Time Zone ID: {time_zone}")
        lines.append("End:")
        lines.append("")
        self.content = "\n".join(lines)

    def add_basin(self, basin_file: BasinFile):
        """Add a basin to the HMS project."""
        lines = [""] + [f"Basin: {basin_file.name}"]
        lines.append(f"     Filename: {basin_file.name}.basin")
        lines.append(f"     Description: {basin_file.header.attrs.get('Description')}")
        lines.append(f'     Last Modified Date: {datetime.now().strftime("%#d %b %Y")}')
        lines.append(f'     Last Modified Time: {datetime.now().strftime("%H:%M:%S")}')
        lines.append("End:")
        lines.append("")
        self.content += "\n".join(lines)
        self.basins.append(basin_file)

    def add_met(self, met_file: MetFile):
        """Add a met to the HMS project."""
        lines = [""] + [f"Precipitation: {met_file.name}"]
        lines.append(f"     Filename: {met_file.name}.met")
        lines.append(f"     Description: {met_file.attrs.get('Description')}")
        lines.append(f'     Last Modified Date: {datetime.now().strftime("%#d %b %Y")}')
        lines.append(f'     Last Modified Time: {datetime.now().strftime("%H:%M:%S")}')
        lines.append("End:")
        self.content += "\n".join(lines)
        self.mets.append(met_file)

    def add_control(self, control_file: ControlFile):
        """Add a control to the HMS project."""
        lines = [""] + [f"Control: {control_file.name}"]
        lines.append(f"     FileName: {control_file.name}.control")
        lines.append(f"     Description: {control_file.attrs.get('Description')}")
        lines.append("End:")
        self.content += "\n".join(lines)
        self.controls.append(control_file)

    @property
    @lru_cache
    def name(self):
        lines = self.content.splitlines()
        if not lines[0].startswith("Project: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Project: ") :]

    def combine_stem_ext(self, ext: str) -> str:
        return f"{self.stem}.{ext}"

    def scan_for_terrain_run_grid_gage_pdata(self):
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
        return {
            "basins": len(self.basins),
            "controls": len(self.controls),
            "mets": len(self.mets),
            "runs": 1 if self.run is not None else None,
            "terrain": 1 if self.terrain is not None else None,
            "pdata": 1 if self.pdata is not None else None,
            "grid": 1 if self.grid is not None else None,
            "gage": 1 if self.gage is not None else None,
            "sqlite": len([basin.sqlite_path for basin in self.basins if os.path.exists(basin.sqlite_path)]),
        }

    def assert_uniform_version(self):
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
        return self.absolute_paths(
            set(
                [gage.attrs["Variant"]["Variant-1"]["DSS File Name"] for gage in self.gage.elements.elements.values()]
                + [
                    grid.attrs["Variant"]["Variant-1"]["DSS File Name"]
                    for grid in self.grid.elements.elements.values()
                    if "Variant" in grid.attrs
                ]
                + [pdata.attrs["DSS File"] for pdata in self.pdata.elements.elements.values()]
            )
        )

    @property
    def result_files(self):
        return self.absolute_paths(
            set(
                [i[1].attrs["Log File"] for i in self.run.elements]
                + [i[1].attrs["DSS File"] for i in self.run.elements]
                + [i[1].attrs["DSS File"].replace(".dss", ".out") for i in self.run.elements]
            )
        )

    def absolute_paths(self, paths):
        return [os.path.join(self.directory, path) for path in paths]

    @property
    def rasters(self):
        files = []
        if self.terrain:
            for terrain in self.terrain.layers:
                files += [os.path.join(terrain["raster_dir"], f) for f in os.listdir(terrain["raster_dir"])]
        files += [grid.attrs["Filename"] for grid in self.grid.elements.elements.values() if "Filename" in grid.attrs]
        return self.absolute_paths(set(files))

    @property
    @lru_cache
    def sqlitedbs(self):
        return [SqliteDB(basin.sqlite_path) for basin in self.basins]


class BasinFile(BaseTextFile):
    def __init__(
        self,
        path: str,
        skip_scans: bool = False,
        client=None,
        bucket=None,
        fiona_aws_session=None,
        read_geom: bool = True,
        new_file: bool = False,
    ):
        if not path.endswith(".basin"):
            raise ValueError(f"invalid extension for Basin file: {path}")
        super().__init__(path, client=client, bucket=bucket, new_file=new_file)

        self.header: BasinHeader = ""
        self.layer_properties: BasinLayerProperties = None
        self.spatial_properties: BasinSpatialProperties = None
        self.schematic_properties: BasinSchematicProperties = None
        self.computation_points: ComputationPoints = None
        self.fiona_aws_session = fiona_aws_session
        self.read_geom = read_geom
        self.name = None
        if not new_file:
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
        for line in self.spatial_properties.content.splitlines():
            if "Coordinate System: " in line:
                return line.split(": ")[1]

    @property
    def crs(self):
        return CRS(self.wkt)

    @property
    def epsg(self):
        return self.crs.to_epsg()

    def compute_ds_from_schematic(self, path: str):
        layers = {}
        for layer in fiona.listlayers(path):
            layers[layer] = gpd.read_file(path, layer=layer)
        gdf = pd.concat(
            [layers[layer] for layer in ["Junction", "Sink", "Reservoir"] if layer in layers], ignore_index=True
        )
        bc = layers["Subbasin_Connectors"]
        bc["start"] = bc.apply(
            lambda x: (
                x.geometry.boundary.geoms[0]
                if isinstance(x.geometry, LineString)
                else x.geometry.geoms[0].boundary.geoms[0]
            ),
            axis=1,
        )
        bc["end"] = bc.apply(
            lambda x: (
                x.geometry.boundary.geoms[1]
                if isinstance(x.geometry, LineString)
                else x.geometry.geoms[0].boundary.geoms[1]
            ),
            axis=1,
        )
        for _, row in layers["Subbasin"].iterrows():
            for _, c in bc.iterrows():
                if row.geometry.contains(c.start):
                    ds = gdf.iloc[gdf.distance(c.end).idxmin()]["name"]
                    break
            layers["Subbasin"].loc[layers["Subbasin"]["name"] == row["name"], "Downstream"] = ds

        reaches = layers["Reach"].copy()
        reaches["geometry"] = reaches.apply(
            lambda x: get_point(x.geometry, -1) if isinstance(x, LineString) else get_point(x.geometry.geoms[0], -1),
            axis=1,
        )

        gdf = pd.concat(
            [layers[layer] for layer in ["Junction", "Sink", "Reservoir"] if layer in layers] + [reaches],
            ignore_index=True,
        )

        jc = layers["Junction_Connectors"]
        jc["start"] = jc.apply(
            lambda x: (
                x.geometry.boundary.geoms[0]
                if isinstance(x.geometry, LineString)
                else x.geometry.geoms[0].boundary.geoms[0]
            ),
            axis=1,
        )
        jc["end"] = jc.apply(
            lambda x: (
                x.geometry.boundary.geoms[1]
                if isinstance(x.geometry, LineString)
                else x.geometry.geoms[0].boundary.geoms[1]
            ),
            axis=1,
        )

        for _, row in layers["Junction"].iterrows():
            ds = None
            candidates = gdf.loc[gdf["name"] != row["name"]]
            for _, c in jc.iterrows():
                if row.geometry.equals(c.start):
                    ds = candidates.loc[candidates.distance(c.end).idxmin(), "name"]
                    break
            if ds is None:
                ds = candidates.loc[candidates.distance(row.geometry).idxmin(), "name"]
            layers["Junction"].loc[layers["Junction"]["name"] == row["name"], "Downstream"] = ds

        gdf = pd.concat(
            [layers[layer] for layer in ["Junction", "Sink", "Reservoir"] if layer in layers], ignore_index=True
        )
        for _, row in layers["Reach"].iterrows():
            start, end = row.geometry.boundary.geoms
            ds = gdf.iloc[gdf.distance(end).idxmin()]["name"]
            layers["Reach"].loc[layers["Reach"]["name"] == row["name"], "Downstream"] = ds

        for layer_name in ["Subbasin", "Junction", "Sink", "Reach"]:
            layer = layers[layer_name]
            if path.endswith(".gpkg"):
                layer.to_file(path, layer=layer_name, driver="GPKG", engine=GPD_WRITE_ENGINE)
            elif path.endswith(".gdb"):
                layer.to_file(path, layer=layer_name, engine=GPD_WRITE_ENGINE, driver="OpenFileGDB")

    @classmethod
    def from_gpkg(cls, gpkg_path: str, basin_file_path: str, version: str = "4.11", update_connections: bool = False):

        inst = cls(basin_file_path, skip_scans=True, read_geom=False, new_file=True)
        if update_connections:
            inst.compute_ds_from_schematic(gpkg_path)

        inst.name = os.path.splitext(os.path.basename(basin_file_path))[0]
        basin_header = DEFAULT_BASIN_HEADER
        basin_header["Version"] = version
        inst.header = BasinHeader(basin_header)
        inst.layer_properties = BasinLayerProperties(DEFAULT_BASIN_LAYER_PROPERTIES)
        gdf = gpd.read_file(gpkg_path, layer="Subbasin", engine="fiona")
        inst.spatial_properties = BasinSpatialProperties(f"Coordinate System: {gdf.crs.to_wkt()}")
        inst.schematic_properties = BasinSchematicProperties({})
        sqlite_path = basin_file_path.replace(".basin", ".sqlite")
        elements = ElementSet()
        for layer in fiona.listlayers(gpkg_path):
            if layer in BASIN_DEFAULTS.keys():
                gdf = gpd.read_file(gpkg_path, layer=BASIN_DEFAULTS[layer]["gpkg_layer"], engine=GPD_WRITE_ENGINE)

                for _, row in gdf.iterrows():
                    attrs = {}
                    for attr, val in BASIN_DEFAULTS[layer]["attrs"].items():
                        if layer in ["Subbasin", "Reach"]:
                            attrs["Canvas X"] = str(row.geometry.centroid.x)
                            attrs["Canvas Y"] = str(row.geometry.centroid.y)
                        else:
                            attrs["Canvas X"] = str(row.geometry.x)
                            attrs["Canvas Y"] = str(row.geometry.y)
                        if attr in gdf.columns:
                            attrs[attr] = str(row[attr])
                        elif attr == "File":
                            attrs[attr] = os.path.basename(sqlite_path)
                        else:
                            attrs[attr] = val
                    elements[row["name"]] = BASIN_DEFAULTS[layer]["element"](row["name"], attrs, row.geometry)
                gdf.to_file(
                    sqlite_path, layer=BASIN_DEFAULTS[layer]["sqlite_layer"], engine=GPD_WRITE_ENGINE, driver="SQLite"
                )
        inst.elements = elements
        # inst.crs = gdf.crs

        return inst

    @classmethod
    def from_gdb(cls, gdb_path: str, basin_file_path: str, version: str = "4.11", update_connections: bool = False):

        inst = cls(basin_file_path, skip_scans=True, read_geom=False, new_file=True)
        if update_connections:
            inst.compute_ds_from_schematic(gdb_path)

        inst.name = os.path.splitext(os.path.basename(basin_file_path))[0]
        basin_header = DEFAULT_BASIN_HEADER
        basin_header["Version"] = version
        inst.header = BasinHeader(basin_header)
        inst.layer_properties = BasinLayerProperties(DEFAULT_BASIN_LAYER_PROPERTIES)
        gdf = gpd.read_file(gdb_path, layer="Subbasin", engine="fiona", driver="OpenFileGDB")
        inst.spatial_properties = BasinSpatialProperties(f"Coordinate System: {gdf.crs.to_wkt()}")
        inst.schematic_properties = BasinSchematicProperties({})
        sqlite_path = basin_file_path.replace(".basin", ".sqlite")
        elements = ElementSet()
        for layer in fiona.listlayers(gdb_path):
            if layer in BASIN_DEFAULTS.keys():
                gdf = gpd.read_file(
                    gdb_path, layer=BASIN_DEFAULTS[layer]["gpkg_layer"], engine=GPD_WRITE_ENGINE, driver="OpenFileGDB"
                )

                for _, row in gdf.iterrows():
                    attrs = {}
                    for attr, val in BASIN_DEFAULTS[layer]["attrs"].items():
                        if layer in ["Subbasin", "Reach"]:
                            attrs["Canvas X"] = str(row.geometry.centroid.x)
                            attrs["Canvas Y"] = str(row.geometry.centroid.y)
                        else:
                            attrs["Canvas X"] = str(row.geometry.x)
                            attrs["Canvas Y"] = str(row.geometry.y)
                        if attr in gdf.columns:
                            attrs[attr] = str(row[attr])
                        elif attr == "File":
                            attrs[attr] = os.path.basename(sqlite_path)
                        else:
                            attrs[attr] = val
                    elements[row["name"]] = BASIN_DEFAULTS[layer]["element"](row["name"], attrs, row.geometry)
                gdf.to_file(
                    sqlite_path, layer=BASIN_DEFAULTS[layer]["sqlite_layer"], engine=GPD_WRITE_ENGINE, driver="SQLite"
                )
        inst.elements = elements
        # inst.crs = gdf.crs

        return inst

    def parse_name(self):
        lines = self.content.splitlines()
        if not lines[0].startswith("Basin: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        self.name = lines[0][len("Basin: ") :]

    def scan_for_headers_and_footers(self):
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

    def serialize(self):
        lines = [f"Basin: {self.name}"] + utils.attrs2list(self.header.attrs) + ["End:"] + [""]
        for _, element in self.elements:
            lines.append(f"{type(element).__name__}: {element.name}")
            lines += utils.attrs2list(element.attrs)
            lines.append("End:")
            lines.append("")
        lines += ["Basin Layer Properties:"] + self.layer_properties.content.splitlines() + ["End:"] + [""]
        lines += ["Basin Spatial Properties:"] + self.spatial_properties.content.splitlines() + ["End:"] + [""]
        lines += ["Basin Schematic Properties:"] + utils.attrs2list(self.schematic_properties.attrs) + ["End:"] + [""]
        if self.computation_points:
            lines += (
                ["Computation Points:"]
                + self.computation_points.content.splitlines()
                + ["End Computation Point: "]
                + [""]
            )
        content = "\n".join(lines)
        self.content = content

    def identify_sqlite(self):
        for line in self.content.splitlines():
            if ".sqlite" in line:
                return line.split("File: ")[1]

    @property
    @lru_cache
    def elements(self):
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
                    geom = sqlite.reach_feats[sqlite.reach_feats["name"] == name].geometry.values[0]
                    if "slope" in sqlite.reach_feats.columns:
                        slope = sqlite.reach_feats[sqlite.reach_feats["name"] == name]["slope"].values[0]
                    else:
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
        return self.elements.get_element_type("Subbasin")

    @property
    @lru_cache
    def reaches(self):
        return self.elements.get_element_type("Reach")

    @property
    @lru_cache
    def junctions(self):
        return self.elements.get_element_type("Junction")

    @property
    @lru_cache
    def reservoirs(self):
        return self.elements.get_element_type("Reservoir")

    @property
    @lru_cache
    def diversions(self):
        return self.elements.get_element_type("Diversion")

    @property
    @lru_cache
    def sinks(self):
        return self.elements.get_element_type("Sink")

    @property
    @lru_cache
    def sources(self):
        return self.elements.get_element_type("Source")

    @property
    @lru_cache
    def gages(self):
        return self.elements.gages

    @property
    @lru_cache
    def drainage_area(self):
        return sum([subbasin.geom.area for subbasin in self.subbasins])

    @property
    @lru_cache
    def reach_miles(self):
        return sum([reach.geom.length for reach in self.reaches])

    def update_initial_baseflow(self, gdf: gpd.GeoDataFrame):
        for subbasin in self.elements.subset(Subbasin):
            try:
                subbasin[1].attrs["GW-2 Initial Flow/Area Ratio"] = str(
                    gdf.loc[gdf.name == subbasin[0], "sampled_value"].values[0]
                )
            except:
                raise IndexError(f"could not find {subbasin[0]} in baseflow data")

    def update_swe_grid_name(self, grid_name: str):
        for subbasin in self.elements.subset(Subbasin):
            subbasin[1].attrs["Snow Water Equivalent Grid Name"] = grid_name

    def ati_meltrate_to_basin_file(self, year: str, event: str, meltrate_function_elements: list):

        for subbasin in self.elements.get_element_type("Subbasin"):

            # update dry meltrate type to ATI function
            subbasin.attrs["Dry Melt Rate Method"] = "ATI Function"

            # remove dry melt rate value
            if "Dry Melt Rate" in subbasin.attrs.keys():
                subbasin.attrs.pop("Dry Melt Rate")

            # add dry melt rate function for each subbasin
            if subbasin.name in meltrate_function_elements:
                subbasin.attrs = utils.insert_after_key(
                    subbasin.attrs,
                    "Melt Rate ATI-Cold Rate Table Name",
                    "Melt Rate ATI-Melt Rate Table Name",
                    f"ATI-MELTRATE_{subbasin.name}_Y{year}_E{event}",
                )
            else:
                subbasin.attrs = utils.insert_after_key(
                    subbasin.attrs,
                    "Melt Rate ATI-Cold Rate Table Name",
                    "Melt Rate ATI-Melt Rate Table Name",
                    f"ATI-MELTRATE_default_Y{year}_E{event}",
                )

    @property
    @lru_cache
    def basin_geom(self):
        return utils.remove_holes(self.feature_2_gdf("Subbasin").make_valid().to_crs(4326).union_all())

    def bbox(self, crs):
        return self.feature_2_gdf("Subbasin").to_crs(crs).total_bounds

    def feature_2_gdf(self, element_type: str) -> gpd.GeoDataFrame:
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
        df_list = []
        for junction in self.junctions:
            us_point = junction.geom
            if "Downstream" not in junction.attrs:
                print(f"Warning no downstream element for junction {junction.name}")
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

    def feature_2_shapefile(self, element_type: str, dest_directory: str = ""):
        if not dest_directory:
            dest_file = os.path.join(
                self.directory,
                f"{self.name}_schematic",
                "schematic_v2",
                f"{element_type}.shp",
            )
        else:
            dest_file = os.path.join(dest_directory, f"{element_type}.shp")
        print(f"writing {dest_file}")
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        if self.elements.get_element_type(element_type):
            self.feature_2_gdf(element_type).to_file(dest_file, engine=GPD_WRITE_ENGINE)

    @property
    @lru_cache
    def hms_schematic_2_gdfs(self) -> dict[gpd.GeoDataFrame]:
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

    def hms_schematic_2_shapefiles(self, dest_directory: str = ""):
        if not dest_directory:
            dest_directory = os.path.join(self.directory, f"{self.name}_schematic", "schematic_v2")
        else:
            dest_directory = os.path.join(dest_directory, f"{self.name}_schematic")
        os.makedirs(dest_directory, exist_ok=True)
        for layer, gdf in self.hms_schematic_2_gdfs().items():
            dest_file = os.path.join(dest_directory, f"{layer}.shp")
            print(f"writing {dest_file}")
            gdf.to_file(dest_file, engine=GPD_WRITE_ENGINE)

    def hms_schematic_2_geopackage(self, dest_directory: str = ""):
        if not dest_directory:
            dest_directory = os.path.join(self.directory, f"{self.name}_schematic", "schematic_v2")
        else:
            dest_directory = os.path.join(dest_directory, f"{self.name}_schematic")
        os.makedirs(dest_directory, exist_ok=True)
        for layer, gdf in self.hms_schematic_2_gdfs().items():
            dest_file = os.path.join(dest_directory, "schematic_v2.gpkg")
            print(f"writing {layer} to {dest_file}")
            gdf.to_file(dest_file, layer=layer, engine=GPD_WRITE_ENGINE)

    def subbasin_bc_lines(self):
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


class MetFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None, new_file: bool = False):
        if not path.endswith(".met"):
            raise ValueError(f"invalid extension for Meteorology file: {path}")
        super().__init__(path, client=client, bucket=bucket, new_file=new_file)
        if not new_file:
            self.scan_for_elements()

    def __repr__(self):
        """Representation of the HMSMetFile class."""
        return f"HMSMetFile({self.path})"

    def serialize(self):
        lines = [f"Meteorology: {self.name}"]
        lines += utils.attrs2list(self.attrs)
        lines.append("End:")
        lines.append("")
        for _, element in self.elements:
            lines.append(f"{type(element).__name__}: {element.name}")
            lines += utils.attrs2list(element.attrs)
            lines.append("End:")
            lines.append("")
        content = "\n".join(lines)
        self.content = content

    @property
    @lru_cache
    def name(self):
        lines = self.content.splitlines()
        if not lines[0].startswith("Meteorology: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Meteorology: ") :]

    def scan_for_elements(self):
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

    def new_met(
        self,
        title: str,
        precipitation_grid_name: str,
        temperature_grid_name: str,
        version: str = "4.11",
        basin_file: BasinFile = None,
        ET: bool = False,
    ):
        if ET:
            if basin_file is None:
                raise ValueError("BasinFile is required for ET")

        lines = []
        lines.append(f"Meteorology: {title}")
        lines.append(f"     Last Modified Date: {datetime.now().strftime('%d %B %Y')}")
        lines.append(f"     Last Modified Time: {datetime.now().strftime('%H:%M:%S')}")
        lines.append(f"     Version: {version}")
        lines.append("     Unit System: English")
        lines.append("     Set Missing Data to Default: Yes")
        lines.append("     Precipitation Method: Gridded Precipitation")
        lines.append("     Air Temperature Method: Grid")
        lines.append("     Atmospheric Pressure Method: None")
        lines.append("     Dew Point Method: None")
        lines.append("     Wind Speed Method: None")
        lines.append("     Shortwave Radiation Method: None")
        lines.append("     Longwave Radiation Method: None")
        lines.append("     Snowmelt Method: None")

        if ET:

            lines.append("     Evapotranspiration Method: Gridded Hamon")
            lines.append(f"     Use Basin Model: {basin_file.name}")

        else:

            lines.append("     Evapotranspiration Method: No Evapotranspiration")

        lines.append("End:")
        lines.append("")
        lines.append("Precip Method Parameters: Gridded Precipitation")
        lines.append(f"     Last Modified Date: {datetime.now().strftime('%d %B %Y')}")
        lines.append(f"     Last Modified Time: {datetime.now().strftime('%H:%M:%S')}")
        lines.append(f"     Precip Grid Name: {precipitation_grid_name}")
        lines.append("     Time Shift Method: NONE")
        lines.append("End:")
        lines.append("")
        lines.append("Air Temperature Method Parameters: Grid")
        lines.append(f"     Last Modified Date: {datetime.now().strftime('%d %B %Y')}")
        lines.append(f"     Last Modified Time: {datetime.now().strftime('%H:%M:%S')}")
        lines.append(f"     Temperature Grid Name: {temperature_grid_name}")
        lines.append("     Time Shift Method: NONE")
        lines.append("End:")
        lines.append("")

        if ET:

            lines.append("Evapotranspiration Method Parameters: Gridded Hamon")
            lines.append(f"     Last Modified Date: {datetime.now().strftime('%d %B %Y')}")
            lines.append(f"     Last Modified Time: {datetime.now().strftime('%H:%M:%S')}")
            lines.append("End:")
            lines.append("")

            for name, _ in basin_file.elements.subset(Subbasin):

                lines.append(f"Subbasin: {name}")
                lines.append(f"     Last Modified Date: {datetime.now().strftime('%d %B %Y')}")
                lines.append(f"     Last Modified Time: {datetime.now().strftime('%H:%M:%S')}")
                lines.append("")
                lines.append("     Begin Et: Hamon")
                lines.append("     Hamon Coefficient: 0.0065")
                lines.append("     End Et:")
                lines.append("End:")
                lines.append("")

        self.content = "\n".join(lines)


class ControlFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None, new_file: bool = False):
        if not path.endswith(".control"):
            raise ValueError(f"invalid extension for Control file: {path}")
        super().__init__(path, client=client, bucket=bucket, new_file=new_file)

    def __repr__(self):
        """Representation of the HMSControlFile class."""
        return f"HMSControlFile({self.path})"

    @property
    @lru_cache
    def name(self):
        lines = self.content.splitlines()
        if not lines[0].startswith("Control: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Control: ") :]

    def serialize(self):
        lines = [f"Control: {self.name}"]
        lines += utils.attrs2list(self.attrs)
        lines.append("End:")
        lines.append("")
        content = "\n".join(lines)
        self.content = content

    def new_control(
        self,
        name: str,
        start_datetime: datetime,
        end_datetime: datetime,
        time_interval: int,
        version: str = "4.11",
        time_zone: str = "America/Denver",
        timezone_offset: str = "-25200000",
    ):

        self.content = f"Control: {name}"
        attrs = CONTROL_DEFAULTS["attrs"]
        attrs["Version"] = version
        attrs["Time Id"] = time_zone
        attrs["Time Zone GMT Offset"] = timezone_offset
        attrs["Start Date"] = start_datetime.strftime("%#d %b %Y")
        attrs["Start Time"] = start_datetime.strftime("%H:%M")
        attrs["End Date"] = end_datetime.strftime("%#d %b %Y")
        attrs["End Time"] = end_datetime.strftime("%H:%M")
        attrs["Time Interval"] = str(time_interval)
        self.attrs = attrs
        self.serialize()


class TerrainFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".terrain"):
            raise ValueError(f"invalid extension for Terrain file: {path}")
        super().__init__(path, client=client, bucket=bucket)
        self.layers = []

        found_first = False
        name, raster_path, vert_units = "", "", ""
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
                        "raster_dir": os.path.dirname(raster_path),
                        "vert_units": vert_units,
                    }
                )
                name, raster_path, vert_units = "", "", ""

            elif line.startswith("Terrain Data: "):
                name = line[len("Terrain Data: ") :]
            elif line.startswith("     Elevation File Name: "):
                raster_path_raw = line[len("     Elevation File Name: ") :]
                raster_path = os.path.join(os.path.dirname(self.path), raster_path_raw.replace("\\", os.sep))
            elif line.startswith("     Vertical Units: "):
                vert_units = line[len("     Vertical Units: ") :]

    def __repr__(self):
        """Representation of the HMSTerrainFile class."""
        return f"HMSTerrainFile({self.path})"

    @property
    @lru_cache
    def name(self):
        return None


class RunFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".run"):
            raise ValueError(f"invalid extension for Run file: {path}")
        super().__init__(path, client=client, bucket=bucket)

    def __repr__(self):
        """Representation of the HMSRunFile class."""
        return f"HMSRunFile({self.path})"

    def runs(self):
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
        return self.runs()

    def add_run(self, name: str, log_file: str, dss_file: str, basin: str, met: str, control: str):
        run = Run(name=name, attrs=RUN_DEFAULTS["attrs"])
        run.attrs["Log File"] = log_file
        run.attrs["DSS File"] = dss_file
        run.attrs["Basin"] = basin
        run.attrs["Precip"] = met
        run.attrs["Control"] = control
        self.elements[name] = run


class PairedDataFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None):
        if not path.endswith(".pdata"):
            raise ValueError(f"invalid extension for Paired Data file: {path}")
        if not os.path.exists(path):
            try:
                response = client.get_object(Bucket=bucket, Key=path)
                self.content = response["Body"].read().decode()
            except Exception as E:
                print(E)
                print("No Paired Data File found: creating empty Paired Data File")
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
        lines = self.content.splitlines()
        if not lines[0].startswith("Paired Data Manager: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Paired Data Manager: ") :]

    def write_table(self, table: Table):
        lines = [""]
        lines.append(f"Table: {table.name}")
        lines.append(f"     Table Type: {table.table_type}")
        lines.append(f'     Last Modified Date: {datetime.now().strftime("%#d %b %Y")}')
        lines.append(f'     Last Modified Time: {datetime.now().strftime("%H:%M:%S")}')
        lines.append(f"     X-Units: {table.x_units}")
        lines.append(f"     Y-Units: {table.y_units}")
        lines.append("     Use External DSS File: NO")
        lines.append(f"     DSS File: {table.dss_file}")
        lines.append(f"     Pathname: {table.pathname}")
        lines.append("     Interpolation: Linear Interpolation")
        lines.append("End:")

        return lines

    def serialize(self):
        lines = [f"Paired Data Manager: {self.name}"]
        lines += utils.attrs2list(self.attrs)
        lines.append("End:")
        lines.append("")
        for _, element in self.elements:
            lines.append(f"{type(element).__name__}: {element.attrs['name']}")
            lines += utils.attrs2list(element.attrs)
            lines.append("End:")
            lines.append("")
        content = "\n".join(lines)
        self.content = content

    def scan_for_tables(self):
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Table: "):
                name = line[len("Table: ") :]
                table_type = lines[i + 1][len("     Table Type: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[f"{name}+{table_type}"] = Table(name, attrs)

    def scan_for_patterns(self):
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Pattern: "):
                name = line[len("Pattern: ") :]
                data_type = lines[i + 1][len("     Data Type: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[f"{name}+{data_type}"] = Pattern(name, attrs)


class SqliteDB:
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

    def delete_layer(self, layer: str):
        if layer not in self.layers:
            raise FileNotFoundError(f"Could not find '{layer}' in {self.path}")
        fiona.remove(self.path, layer=layer)

    def gdf_to_layer(self, gdf: gpd.GeoDataFrame, layer: str):
        gdf.to_file(self.path, layer=layer, engine=GPD_WRITE_ENGINE)

    def shapefile_to_layer(self, shapefile: str, layer: str):
        gdf = gpd.read_file(shapefile)
        gdf.to_file(self.path, layer=layer, engine=GPD_WRITE_ENGINE)

    def geojson_to_layer(self, shapefile: str, layer: str):
        gdf = gpd.read_file(shapefile)
        gdf.to_file(self.path, layer=layer, engine=GPD_WRITE_ENGINE)

    def layer_to_gdf(self, layer: str):
        return gpd.read_file(self.path, layer=layer)

    def layer_to_shapefile(self, layer: str, output_directory: str):
        gdf = gpd.read_file(self.path, layer=layer)
        gdf.to_file(os.path.join(output_directory, layer + ".shp"), engine=GPD_WRITE_ENGINE)

    def dump_to_shapefiles(self, output_directory: str):
        for layer in self.layers:
            gdf = gpd.read_file(self.path, layer=layer)
            gdf.to_file(os.path.join(output_directory, layer + ".shp"), engine=GPD_WRITE_ENGINE)

    def layer_to_geojson(self, layer: str, output_directory: str):
        gdf = gpd.read_file(self.path, layer=layer)
        gdf.to_file(os.path.join(output_directory, layer + ".geojson"), engine=GPD_WRITE_ENGINE)

    def dump_to_geojson(self, output_directory: str):
        for layer in self.layers:
            gdf = gpd.read_file(self.path, layer=layer)
            gdf.to_file(os.path.join(output_directory, layer + ".geojson"), engine=GPD_WRITE_ENGINE)


class GridFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None, new_file: bool = False):
        if not path.endswith(".grid"):
            raise ValueError(f"invalid extension for Grid file: {path}")
        super().__init__(path, client=client, bucket=bucket, new_file=new_file)
        self.elements = ElementSet()
        if not new_file:
            self.scan_for_grids()

    def __repr__(self):
        """Representation of the HMSGridFile class."""
        return f"HMSGridFile({self.path})"

    @property
    @lru_cache
    def name(self):
        lines = self.content.splitlines()
        if not lines[0].startswith("Grid Manager: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Grid Manager: ") :]

    def scan_for_grids(self):
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Grid: "):
                name = line[len("Grid: ") :]
                grid_type = lines[i + 1][len("     Grid Type: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[f"{name}+{grid_type}"] = Grid(f"{name}+{grid_type}", attrs)

    def serialize(self):
        lines = [f"Grid Manager: {self.name}"]
        lines += utils.attrs2list(self.attrs)
        lines.append("End:")
        lines.append("")
        for _, element in self.elements:
            lines.append(f"{type(element).__name__}: {element.attrs['name']}")
            lines += utils.attrs2list(element.attrs)
            lines.append("End:")
            lines.append("")
        content = "\n".join(lines)
        self.content = content

    def remove_grid_type(self, grid_types: list[str]):
        new_elements = ElementSet()
        for name, g in self.elements.elements.items():
            if g.attrs["Grid Type"] not in grid_types:
                new_elements[name] = g
        self.elements = new_elements

    @property
    @lru_cache
    def grids(self):
        return self.elements.get_element_type("Grid")

    def new_grid_manager(self, title: str, version: str = "4.11"):
        lines = [f"Grid Manager: {title}"]
        lines.append(f"     Version: {version}")
        lines.append("     Filepath Separator: \\")
        lines.append("End:")
        lines.append("")
        self.content = "\n".join(lines)
        self.parse_header()

    def add_grid(self, name: str, grid_type: str, dss_file: str, dss_path: str):
        """Add a new grid to the grid file."""
        if grid_type not in ["Precipitation", "Temperature", "Snow Water Equivalent"]:
            raise ValueError(
                f"Unrecognized grid type. Only Precipitation, Temperature, and Snow Water Equivalent are supported at this time. Recieved {grid_type}"
            )
        grid = Grid(name=f"{name}+{grid_type}", attrs=GRID_DEFAULTS["attrs"])
        # grid.attrs["Grid"] = name
        grid.attrs["Grid Type"] = grid_type
        grid.attrs["  DSS File Name"] = dss_file
        grid.attrs["  DSS Pathname"] = dss_path
        self.elements[f"{name}+{grid_type}"] = grid


class GageFile(BaseTextFile):
    def __init__(self, path: str, client=None, bucket=None, new_file: bool = False):
        if not path.endswith(".gage"):
            raise ValueError(f"invalid extension for Gage file: {path}")
        super().__init__(path, client=client, bucket=bucket, new_file=new_file)
        self.elements = ElementSet()
        if not new_file:
            self.scan_for_gages()

    def __repr__(self):
        """Representation of the HMSGageFile class."""
        return f"HMSGageFile({self.path})"

    @property
    @lru_cache
    def name(self):
        lines = self.content.splitlines()
        if not lines[0].startswith("Gage Manager: "):
            raise ValueError(f"unexpected first line: {lines[0]}")
        return lines[0][len("Gage Manager: ") :]

    def scan_for_gages(self):
        lines = self.content.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("Gage: "):
                name = line[len("Gage: ") :]
                attrs = utils.parse_attrs(lines[i + 1 :])
                self.elements[name] = Gage(name, attrs)

    def serialize(self):
        lines = [f"Gage Manager: {self.name}"]
        lines += utils.attrs2list(self.attrs)
        lines.append("End:")
        lines.append("")
        for _, element in self.elements:
            lines.append(f"{type(element).__name__}: {element.name.split('+')[0]}")
            lines += utils.attrs2list(element.attrs)
            lines.append("End:")
            lines.append("")
        content = "\n".join(lines)
        self.content = content

    @property
    @lru_cache
    def gages(self):
        return self.elements.get_element_type("Gage")

    def new_gage_manager(self, title: str, version: str = "4.11"):
        lines = [f"Gage Manager: {title}"]
        lines.append(f"     Version: {version}")
        lines.append("     Filepath Separator: \\")
        lines.append("End:")
        lines.append("")
        self.content = "\n".join(lines)
        self.parse_header()

    def add_gage(
        self, name: str, dss_file: str, dss_path: str, start_date: datetime, end_date: datetime, gage_type: str = "Flow"
    ):
        """
        Addd a gage to the gage manager.

        //part_b/part_c/part_d/part_e/par_f/
        """
        if gage_type != "Flow":
            raise ValueError(f"Only Flow gages are supported at this time. Recieved: {gage_type}")

        gage = Gage(name=name, attrs=GAGE_DEFAULTS["attrs"])
        gage.attrs["Gage Type"] = gage_type
        gage.attrs["  DSS File Name"] = dss_file
        gage.attrs["  DSS Pathname"] = dss_path
        gage.attrs["  Start Time"] = start_date.strftime("%d %B %Y, %H:%M")
        gage.attrs["  End Time"] = end_date.strftime("%d %B %Y, %H:%M")
        self.elements[name] = gage
