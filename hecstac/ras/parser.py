import datetime
import logging
import math
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import Asset
from rashdf import RasGeomHdf, RasHdf, RasPlanHdf
from shapely import LineString, MultiPolygon, Point, Polygon, make_valid, union_all
from shapely.ops import unary_union

from hecstac.ras.utils import (
    data_pairs_from_text_block,
    delimited_pairs_to_lists,
    search_contents,
    text_block_from_start_end_str,
    text_block_from_start_str_length,
    text_block_from_start_str_to_empty_line,
)


def name_from_suffix(fpath: str, suffix: str) -> str:
    """Generate a name by appending a suffix to the file stem."""
    return f"{Path(fpath).stem}.{suffix}"


class River:

    def __init__(self, river: str, reaches: list[str] = []):
        self.river = river
        self.reaches = reaches


class XS:
    """HEC-RAS Cross Section."""

    def __init__(self, ras_data: list[str], river_reach: str, river: str, reach: str, crs: str):
        self.ras_data = ras_data
        self.crs = crs
        self.river = river
        self.reach = reach
        self.river_reach = river_reach
        self.river_reach_rs = f"{river} {reach} {self.river_station}"
        self._is_interpolated: bool | None = None

    def split_xs_header(self, position: int):
        """
        Split cross section header.

        Example: Type RM Length L Ch R = 1 ,83554.  ,237.02,192.39,113.07.
        """
        header = search_contents(self.ras_data, "Type RM Length L Ch R ", expect_one=True)

        return header.split(",")[position]

    @property
    def river_station(self) -> float:
        """Cross section river station."""
        return float(self.split_xs_header(1).replace("*", ""))

    @property
    def left_reach_length(self) -> float:
        """Cross section left reach length."""
        return float(self.split_xs_header(2))

    @property
    def channel_reach_length(self) -> float:
        """Cross section channel reach length."""
        return float(self.split_xs_header(3))

    @property
    def right_reach_length(self) -> float:
        """Cross section right reach length."""
        return float(self.split_xs_header(4))

    @property
    def number_of_coords(self) -> int:
        """Number of coordinates in cross section."""
        try:
            return int(search_contents(self.ras_data, "XS GIS Cut Line", expect_one=True))
        except ValueError:
            return 0
            # raise NotGeoreferencedError(f"No coordinates found for cross section: {self.river_reach_rs} ")

    @property
    def thalweg(self) -> float | None:
        """Cross section thalweg elevation."""
        if self.station_elevation_points:
            _, y = list(zip(*self.station_elevation_points))
            return min(y)

    @property
    def xs_max_elevation(self) -> float | None:
        """Cross section maximum elevation."""
        if self.station_elevation_points:
            _, y = list(zip(*self.station_elevation_points))
            return max(y)

    @property
    def coords(self) -> list[tuple[float, float]] | None:
        """Cross section coordinates."""
        lines = text_block_from_start_str_length(
            f"XS GIS Cut Line={self.number_of_coords}",
            math.ceil(self.number_of_coords / 2),
            self.ras_data,
        )
        if lines:
            return data_pairs_from_text_block(lines, 32)

    @property
    def number_of_station_elevation_points(self) -> int:
        """Number of station elevation points."""
        return int(search_contents(self.ras_data, "#Sta/Elev", expect_one=True))

    @property
    def station_elevation_points(self) -> list[tuple[float, float]] | None:
        """Station elevation points."""
        try:
            lines = text_block_from_start_str_length(
                f"#Sta/Elev= {self.number_of_station_elevation_points} ",
                math.ceil(self.number_of_station_elevation_points / 5),
                self.ras_data,
            )
            return data_pairs_from_text_block(lines, 16)
        except ValueError:
            return None

    @property
    def bank_stations(self) -> list[str]:
        """Bank stations."""
        return search_contents(self.ras_data, "Bank Sta", expect_one=True).split(",")

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Cross section geodataframe."""
        return gpd.GeoDataFrame(
            {
                "geometry": [LineString(self.coords)],
                "river": [self.river],
                "reach": [self.reach],
                "river_reach": [self.river_reach],
                "river_station": [self.river_station],
                "river_reach_rs": [self.river_reach_rs],
                "thalweg": [self.thalweg],
                "xs_max_elevation": [self.xs_max_elevation],
                "left_reach_length": [self.left_reach_length],
                "right_reach_length": [self.right_reach_length],
                "channel_reach_length": [self.channel_reach_length],
                "ras_data": ["\n".join(self.ras_data)],
                "station_elevation_points": [self.station_elevation_points],
                "bank_stations": [self.bank_stations],
                "number_of_station_elevation_points": [self.number_of_station_elevation_points],
                "number_of_coords": [self.number_of_coords],
                # "coords": [self.coords],
            },
            crs=self.crs,
            geometry="geometry",
        )

    @property
    def n_subdivisions(self) -> int:
        """Get the number of subdivisions (defined by manning's n)."""
        return int(search_contents(self.ras_data, "#Mann", expect_one=True).split(",")[0])

    @property
    def subdivision_type(self) -> int:
        """Get the subdivision type.

        -1 seems to indicate horizontally-varied n.  0 seems to indicate subdivisions by LOB, channel, ROB.
        """
        return int(search_contents(self.ras_data, "#Mann", expect_one=True).split(",")[1])

    @property
    def subdivisions(self) -> tuple[list[float], list[float]]:
        """Get the stations corresponding to subdivision breaks, along with their roughness."""
        try:
            header = [l for l in self.ras_data if l.startswith("#Mann")][0]
            lines = text_block_from_start_str_length(
                header,
                math.ceil(self.n_subdivisions / 3),
                self.ras_data,
            )

            return delimited_pairs_to_lists(lines)
        except ValueError:
            return None

    @property
    def is_interpolated(self) -> bool:
        if self._is_interpolated == None:
            self._is_interpolated = "*" in self.split_xs_header(1)
        return self._is_interpolated

    def wse_intersection_pts(self, wse: float) -> list[tuple[float, float]]:
        """Find where the cross-section terrain intersects the water-surface elevation."""
        section_pts = self.station_elevation_points
        intersection_pts = []

        # Iterate through all pairs of points and find any points where the line would cross the wse
        for i in range(len(section_pts) - 1):
            p1 = section_pts[i]
            p2 = section_pts[i + 1]

            if p1[1] > wse and p2[1] > wse:  # No intesection
                continue
            elif p1[1] < wse and p2[1] < wse:  # Both below wse
                continue

            # Define line
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - (m * p1[0])

            # Find intersection point with Cramer's rule
            determinant = lambda a, b: (a[0] * b[1]) - (a[1] * b[0])
            div = determinant((1, 1), (-m, 0))
            tmp_y = determinant((b, wse), (-m, 0)) / div
            tmp_x = determinant((1, 1), (b, wse)) / div

            intersection_pts.append((tmp_x, tmp_y))
        return intersection_pts

    def get_wetted_perimeter(self, wse: float, start: float = None, stop: float = None) -> float:
        """Get the hydraulic radius of the cross-section at a given WSE."""
        df = pd.DataFrame(self.station_elevation_points, columns=["x", "y"])
        df = pd.concat([df, pd.DataFrame(self.wse_intersection_pts(wse), columns=["x", "y"])])
        if start is not None:
            df = df[df["x"] >= start]
        if stop is not None:
            df = df[df["x"] <= stop]
        df = df.sort_values("x", ascending=True)
        df = df[df["y"] <= wse]
        if len(df) == 0:
            return 0
        df["dx"] = df["x"].diff(-1)
        df["dy"] = df["y"].diff(-1)
        df["d"] = ((df["x"] ** 2) + (df["y"] ** 2)) ** (0.5)

        return df["d"].cumsum().values[0]

    def get_flow_area(self, wse: float, start: float = None, stop: float = None) -> float:
        """Get the flow area of the cross-section at a given WSE."""
        df = pd.DataFrame(self.station_elevation_points, columns=["x", "y"])
        df = pd.concat([df, pd.DataFrame(self.wse_intersection_pts(wse), columns=["x", "y"])])
        if start is not None:
            df = df[df["x"] >= start]
        if stop is not None:
            df = df[df["x"] <= stop]
        df = df.sort_values("x", ascending=True)
        df = df[df["y"] <= wse]
        if len(df) == 0:
            return 0
        df["d"] = wse - df["y"]  # depth
        df["d2"] = df["d"].shift(-1)
        df["x2"] = df["x"].shift(-1)
        df["a"] = ((df["d"] + df["d2"]) / 2) * (df["x2"] - df["x"])  # area of a trapezoid

        return df["a"].cumsum().values[0]

    def get_mannings_discharge(self, wse: float, slope: float, units: str) -> float:
        """Calculate the discharge of the cross-section according to manning's equation."""
        q = 0
        stations, mannings = self.subdivisions
        slope = slope**0.5  # pre-process slope for efficiency
        for i in range(self.n_subdivisions - 1):
            start = stations[i]
            stop = stations[i + 1]
            n = mannings[i]
            area = self.get_flow_area(wse, start, stop)
            if area == 0:
                continue
            perimeter = self.get_wetted_perimeter(wse, start, stop)
            rh = area / perimeter
            tmp_q = (1 / n) * area * (rh ** (2 / 3)) * slope
            if units == "english":
                tmp_q *= 1.49
            q += (1 / n) * area * (rh ** (2 / 3)) * slope
        return q


class StructureType(Enum):
    XS = 1
    CULVERT = 2
    BRIDGE = 3
    MULTIPLE_OPENING = 4
    INLINE_STRUCTURE = 5
    LATERAL_STRUCTURE = 6


class Structure:
    """Structure."""

    def __init__(
        self,
        ras_data: list[str],
        river_reach: str,
        river: str,
        reach: str,
        crs: str,
        us_xs: XS,
    ):
        self.ras_data = ras_data
        self.crs = crs
        self.river = river
        self.reach = reach
        self.river_reach = river_reach
        self.river_reach_rs = f"{river} {reach} {self.river_station}"
        self.us_xs = us_xs

    def split_structure_header(self, position: int) -> str:
        """
        Split Structure header.

        Example: Type RM Length L Ch R = 3 ,83554.  ,237.02,192.39,113.07.
        """
        header = search_contents(self.ras_data, "Type RM Length L Ch R ", expect_one=True)

        return header.split(",")[position]

    @property
    def river_station(self) -> float:
        """Structure river station."""
        return float(self.split_structure_header(1))

    @property
    def type(self) -> StructureType:
        """Structure type."""
        return StructureType(int(self.split_structure_header(0)))

    def structure_data(self, position: int) -> str | int:
        """Structure data."""
        if self.type in [
            StructureType.XS,
            StructureType.CULVERT,
            StructureType.BRIDGE,
            StructureType.MULTIPLE_OPENING,
        ]:  # 1 = Cross Section, 2 = Culvert, 3 = Bridge, 4 = Multiple Opening
            data = text_block_from_start_str_length(
                "Deck Dist Width WeirC Skew NumUp NumDn MinLoCord MaxHiCord MaxSubmerge Is_Ogee",
                1,
                self.ras_data,
            )
            return data[0].split(",")[position]
        elif self.type == StructureType.INLINE_STRUCTURE:  # 5 = Inline Structure
            data = text_block_from_start_str_length(
                "IW Dist,WD,Coef,Skew,MaxSub,Min_El,Is_Ogee,SpillHt,DesHd",
                1,
                self.ras_data,
            )
            return data[0].split(",")[position]
        elif self.type == StructureType.LATERAL_STRUCTURE:  # 6 = Lateral Structure
            return 0

    @property
    def distance(self) -> float:
        """Distance to upstream cross section."""
        return float(self.structure_data(0))

    @property
    def width(self) -> float:
        """Structure width."""
        # TODO check units of the RAS model
        return float(self.structure_data(1))

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Structure geodataframe."""
        return gpd.GeoDataFrame(
            {
                "geometry": [LineString(self.us_xs.coords).offset_curve(self.distance)],
                "river": [self.river],
                "reach": [self.reach],
                "river_reach": [self.river_reach],
                "river_station": [self.river_station],
                "river_reach_rs": [self.river_reach_rs],
                "type": [self.type],
                "distance": [self.distance],
                "width": [self.width],
                "ras_data": ["\n".join(self.ras_data)],
            },
            crs=self.crs,
            geometry="geometry",
        )


class Reach:
    """HEC-RAS River Reach."""

    def __init__(self, ras_data: list[str], river_reach: str, crs: str):
        reach_lines = text_block_from_start_end_str(f"River Reach={river_reach}", ["River Reach"], ras_data, -1)
        self.ras_data = reach_lines
        self.crs = crs
        self.river_reach = river_reach
        self.river = river_reach.split(",")[0].rstrip()
        self.reach = river_reach.split(",")[1].rstrip()

        us_connection: str = None
        ds_connection: str = None

    @property
    def us_xs(self) -> "XS":
        """Upstream cross section."""
        return self.cross_sections[
            self.xs_gdf.loc[
                self.xs_gdf["river_station"] == self.xs_gdf["river_station"].max(),
                "river_reach_rs",
            ][0]
        ]

    @property
    def ds_xs(self) -> "XS":
        """Downstream cross section."""
        return self.cross_sections[
            self.xs_gdf.loc[
                self.xs_gdf["river_station"] == self.xs_gdf["river_station"].min(),
                "river_reach_rs",
            ][0]
        ]

    @property
    def number_of_cross_sections(self) -> int:
        """Number of cross sections."""
        return len(self.cross_sections)

    @property
    def number_of_coords(self) -> int:
        """Number of coordinates in reach."""
        return int(search_contents(self.ras_data, "Reach XY"))

    @property
    def coords(self) -> list[tuple[float, float]]:
        """Reach coordinates."""
        lines = text_block_from_start_str_length(
            f"Reach XY= {self.number_of_coords} ",
            math.ceil(self.number_of_coords / 2),
            self.ras_data,
        )
        return data_pairs_from_text_block(lines, 32)

    @property
    def reach_nodes(self) -> list[str]:
        """Reach nodes."""
        return search_contents(self.ras_data, "Type RM Length L Ch R ", expect_one=False)

    @property
    def cross_sections(self) -> dict[str, "XS"]:
        """Cross sections."""
        cross_sections = {}
        for header in self.reach_nodes:
            type, _, _, _, _ = header.split(",")[:5]
            if int(type) != 1:
                continue
            xs_lines = text_block_from_start_end_str(
                f"Type RM Length L Ch R ={header}",
                ["Type RM Length L Ch R", "River Reach"],
                self.ras_data,
            )
            cross_section = XS(xs_lines, self.river_reach, self.river, self.reach, self.crs)
            cross_sections[cross_section.river_reach_rs] = cross_section

        return cross_sections

    @property
    def structures(self) -> dict[str, "Structure"]:
        """Structures."""
        structures = {}
        for header in self.reach_nodes:
            type, _, _, _, _ = header.split(",")[:5]
            if int(type) == 1:
                xs_lines = text_block_from_start_end_str(
                    f"Type RM Length L Ch R ={header}",
                    ["Type RM Length L Ch R", "River Reach"],
                    self.ras_data,
                )
                cross_section = XS(xs_lines, self.river_reach, self.river, self.reach, self.crs)
                continue
            elif int(type) in [2, 3, 4, 5, 6]:  # culvert or bridge or multiple openeing
                structure_lines = text_block_from_start_end_str(
                    f"Type RM Length L Ch R ={header}",
                    ["Type RM Length L Ch R", "River Reach"],
                    self.ras_data,
                )
            else:
                raise TypeError(
                    f"Unsupported structure type: {int(type)}. Supported structure types are 2, 3, 4, 5, and 6 corresponding to culvert, \
                        bridge, multiple openeing, inline structure, lateral structure, respectively"
                )

            structure = Structure(
                structure_lines,
                self.river_reach,
                self.river,
                self.reach,
                self.crs,
                cross_section,
            )
            structures[structure.river_reach_rs] = structure

        return structures

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Reach geodataframe."""
        return gpd.GeoDataFrame(
            {
                "geometry": [LineString(self.coords)],
                "river": [self.river],
                "reach": [self.reach],
                "river_reach": [self.river_reach],
                # "number_of_coords": [self.number_of_coords],
                # "coords": [self.coords],
                "ras_data": ["\n".join(self.ras_data)],
            },
            crs=self.crs,
            geometry="geometry",
        )

    @property
    def xs_gdf(self) -> gpd.GeoDataFrame:
        """Cross section geodataframe."""
        return pd.concat([xs.gdf for xs in self.cross_sections.values()])

    @property
    def structures_gdf(self) -> gpd.GeoDataFrame:
        """Structures geodataframe."""
        return pd.concat([structure.gdf for structure in self.structures.values()])


class Junction:
    """HEC-RAS Junction."""

    def __init__(self, ras_data: list[str], junct: str, crs: str):
        self.crs = crs
        self.name = junct
        self.ras_data = text_block_from_start_str_to_empty_line(f"Junct Name={junct}", ras_data)

    def split_lines(self, lines: list[str], token: str, idx: int) -> list[str]:
        """Split lines."""
        return list(map(lambda line: line.split(token)[idx].rstrip(), lines))

    @property
    def x(self) -> float:
        """Junction x coordinate."""
        return float(self.split_lines([search_contents(self.ras_data, "Junct X Y & Text X Y")], ",", 0))

    @property
    def y(self):
        """Junction y coordinate."""
        return float(self.split_lines([search_contents(self.ras_data, "Junct X Y & Text X Y")], ",", 1))

    @property
    def point(self) -> Point:
        """Junction point."""
        return Point(self.x, self.y)

    @property
    def upstream_rivers(self) -> str:
        """Upstream rivers."""
        return ",".join(
            self.split_lines(
                search_contents(self.ras_data, "Up River,Reach", expect_one=False),
                ",",
                0,
            )
        )

    @property
    def downstream_rivers(self) -> str:
        """Downstream rivers."""
        return ",".join(
            self.split_lines(
                search_contents(self.ras_data, "Dn River,Reach", expect_one=False),
                ",",
                0,
            )
        )

    @property
    def upstream_reaches(self) -> str:
        """Upstream reaches."""
        return ",".join(
            self.split_lines(
                search_contents(self.ras_data, "Up River,Reach", expect_one=False),
                ",",
                1,
            )
        )

    @property
    def downstream_reaches(self) -> str:
        """Downstream reaches."""
        return ",".join(
            self.split_lines(
                search_contents(self.ras_data, "Dn River,Reach", expect_one=False),
                ",",
                1,
            )
        )

    @property
    def junction_lengths(self) -> str:
        """Junction lengths."""
        return ",".join(self.split_lines(search_contents(self.ras_data, "Junc L&A", expect_one=False), ",", 0))

    @property
    def gdf(self):
        """Junction geodataframe."""
        return gpd.GeoDataFrame(
            {
                "geometry": [self.point],
                "junction_lengths": [self.junction_lengths],
                "us_rivers": [self.upstream_rivers],
                "ds_rivers": [self.downstream_rivers],
                "us_reaches": [self.upstream_reaches],
                "ds_reaches": [self.downstream_reaches],
                "ras_data": ["\n".join(self.ras_data)],
            },
            geometry="geometry",
            crs=self.crs,
        )


class StorageArea:

    def __init__(self, ras_data: list[str], crs: str):
        self.crs = crs
        self.ras_data = ras_data
        # TODO: Implement this


class Connection:

    def __init__(self, ras_data: list[str], crs: str):
        self.crs = crs
        self.ras_data = ras_data
        # TODO: Implement this


class ProjectFile:
    """HEC-RAS Project file."""

    def __init__(self, fpath):
        # TODO: Compare with HMS implementation
        self.fpath = fpath
        with open(fpath, "r") as f:
            self.file_lines = f.readlines()

    @property
    def project_title(self) -> str:
        return search_contents(self.file_lines, "Proj Title")

    @property
    def project_description(self) -> str:
        return search_contents(self.file_lines, "Model Description", token=":", require_one=False)

    @property
    def project_status(self) -> str:
        return search_contents(self.file_lines, "Status of Model", token=":", require_one=False)

    @property
    def project_units(self) -> str | None:
        for line in self.file_lines:
            if "Units" in line:
                return " ".join(line.split(" ")[:-1])

    @property
    def plan_current(self) -> str | None:
        try:
            suffix = search_contents(self.file_lines, "Current Plan", expect_one=True, require_one=False)
            return self.name_from_suffix(suffix)
        except Exception:
            logging.warning("Ras model has no current plan")
            return None

    @property
    def ras_version(self) -> str | None:
        version = search_contents(self.file_lines, "Program Version", token="=", expect_one=False, require_one=False)
        if version == []:
            version = search_contents(
                self.file_lines, "Program and Version", token=":", expect_one=False, require_one=False
            )
        if version == []:
            logging.warning("Unable to parse project version")
            return "N/A"
        else:
            return version[0]

    @property
    def plan_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Plan File", expect_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]

    @property
    def geometry_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Geom File", expect_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]

    @property
    def steady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Flow File", expect_one=False, require_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]

    @property
    def quasi_unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "QuasiSteady File", expect_one=False, require_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]

    @property
    def unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Unsteady File", expect_one=False, require_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]


class PlanFile:
    """HEC-RAS Plan file asset."""

    def __init__(self, fpath):
        # TODO: Compare with HMS implementation
        self.fpath = fpath
        with open(fpath, "r") as f:
            self.file_lines = f.readlines()

    @property
    def plan_title(self) -> str:
        return search_contents(self.file_lines, "Plan Title")

    @property
    def plan_version(self) -> str:
        return search_contents(self.file_lines, "Program Version")

    @property
    def geometry_file(self) -> str:
        suffix = search_contents(self.file_lines, "Geom File", expect_one=True)
        return name_from_suffix(self.fpath, suffix)

    @property
    def flow_file(self) -> str:
        suffix = search_contents(self.file_lines, "Flow File", expect_one=True)
        return name_from_suffix(self.fpath, suffix)

    @property
    def short_identifier(self) -> str:
        return search_contents(self.file_lines, "Short Identifier", expect_one=True)

    @property
    def breach_locations(self) -> dict:
        """
        example file line:
        Breach Loc=                ,                ,        ,True,HH_DamEmbankment
        """
        breach_dict = {}
        matches = search_contents(self.file_lines, "Breach Loc", expect_one=False, require_one=False)
        for line in matches:
            parts = line.split(",")
            if len(parts) >= 4:
                key = parts[4].strip()
                breach_dict[key] = eval(parts[3].strip())
        logging.debug(f"breach_dict {breach_dict}")
        return breach_dict


class GeometryFile:
    """HEC-RAS Geometry file asset."""

    def __init__(self, fpath, crs):
        # TODO: Compare with HMS implementation
        self.fpath = fpath
        self.crs = crs
        with open(fpath, "r") as f:
            self.file_lines = f.readlines()

    @property
    def geom_title(self) -> str:
        return search_contents(self.file_lines, "Geom Title")

    @property
    def geom_version(self) -> str:
        return search_contents(self.file_lines, "Program Version")

    @property
    def rivers(self) -> dict[str, "River"]:
        """A dictionary of river_name: River (class) for the rivers contained in the HEC-RAS geometry file."""
        tmp_rivers = defaultdict(list)
        for reach in self.reaches.values():  # First, group all reaches into their respective rivers
            tmp_rivers[reach.river].append(reach.reach)
        for (
            river,
            reaches,
        ) in tmp_rivers.items():  # Then, create a River object for each river
            tmp_rivers[river] = River(river, reaches)
        return tmp_rivers

    @property
    def reaches(self) -> dict[str, "Reach"]:
        """A dictionary of the reaches contained in the HEC-RAS geometry file."""
        river_reaches = search_contents(self.file_lines, "River Reach", expect_one=False, require_one=False)
        return {river_reach: Reach(self.file_lines, river_reach, self.crs) for river_reach in river_reaches}

    @property
    def junctions(self) -> dict[str, "Junction"]:
        """A dictionary of the junctions contained in the HEC-RAS geometry file."""
        juncts = search_contents(self.file_lines, "Junct Name", expect_one=False)
        return {junction: Junction(self.file_lines, junction, self.crs) for junction in juncts}

    @property
    def cross_sections(self) -> dict[str, "XS"]:
        """A dictionary of all the cross sections contained in the HEC-RAS geometry file."""
        cross_sections = {}
        for reach in self.reaches.values():
            cross_sections.update(reach.cross_sections)
        return cross_sections

    @property
    def structures(self) -> dict[str, "Structure"]:
        """A dictionary of the structures contained in the HEC-RAS geometry file."""
        structures = {}
        for reach in self.reaches.values():
            structures.update(reach.structures)
        return structures

    @property
    def storage_areas(self) -> dict[str, "StorageArea"]:
        """A dictionary of the storage areas contained in the HEC-RAS geometry file."""
        areas = search_contents(self.file_lines, "Storage Area", expect_one=False)
        return {a: StorageArea(a, self.crs) for a in areas}

    @property
    def connections(self) -> dict[str, "Connection"]:
        """A dictionary of the SA/2D connections contained in the HEC-RAS geometry file."""
        connections = search_contents(self.file_lines, "Connection", expect_one=False)
        return {c: Connection(c, self.crs) for c in connections}

    @property
    def datetimes(self) -> list[datetime.datetime]:
        """Get the latest node last updated entry for this geometry"""
        dts = search_contents(self.file_lines, "Node Last Edited Time", expect_one=False)
        if len(dts) >= 1:
            try:
                return [datetime.datetime.strptime(d, "%b/%d/%Y %H:%M:%S") for d in dts]
            except ValueError:
                return []
        else:
            return []

    @property
    def has_2d(self) -> bool:
        """Check if RAS geometry has any 2D areas"""
        for line in self.file_lines:
            if line.startswith("Storage Area Is2D=") and int(line[len("Storage Area Is2D=") :].strip()) in (1, -1):
                # RAS mostly uses "-1" to indicate True and "0" to indicate False. Checking for "1" also here.
                return True
        return False

    @property
    def has_1d(self) -> bool:
        """Check if RAS geometry has any 1D components"""
        return len(self.cross_sections) > 0

    @property
    def concave_hull(self) -> Polygon:
        """Compute and return the concave hull (polygon) for cross sections."""
        polygons = []
        if self.cross_sections:
            xs_gdf = pd.concat([xs.gdf for xs in self.cross_sections.values()], ignore_index=True)
            # logging.info(xs_gdf)
            for river_reach in xs_gdf["river_reach"].unique():
                logging.info(f"river reach: {river_reach}")
                xs_subset: gpd.GeoSeries = xs_gdf[xs_gdf["river_reach"] == river_reach]
                logging.info(xs_subset)
                points = xs_subset.boundary.explode(index_parts=True).unstack()
                points_last_xs = [Point(coord) for coord in xs_subset["geometry"].iloc[-1].coords]
                points_first_xs = [Point(coord) for coord in xs_subset["geometry"].iloc[0].coords[::-1]]
                polygon = Polygon(points_first_xs + list(points[0]) + points_last_xs + list(points[1])[::-1])
                logging.info("got polygon")
                if isinstance(polygon, MultiPolygon):
                    polygons += list(polygon.geoms)
                else:
                    polygons.append(polygon)
            if len(self.junctions) > 0:
                for junction in self.junctions.values():
                    for _, j in junction.gdf.iterrows():
                        polygons.append(self.junction_hull(xs_gdf, j))
            out_hull = union_all([make_valid(p) for p in polygons])
            return out_hull
        else:
            raise ValueError(f"No cross sections found for {self.fpath}. Cannot calculate geometry")

    def junction_hull(self, xs_gdf: gpd.GeoDataFrame, junction: gpd.GeoSeries) -> Polygon:
        """Compute and return the concave hull (polygon) for a juction."""
        junction_xs = self.determine_junction_xs(xs_gdf, junction)

        junction_xs["start"] = junction_xs.apply(lambda row: row.geometry.boundary.geoms[0], axis=1)
        junction_xs["end"] = junction_xs.apply(lambda row: row.geometry.boundary.geoms[1], axis=1)
        junction_xs["to_line"] = junction_xs.apply(lambda row: self.determine_xs_order(row, junction_xs), axis=1)

        coords = []
        first_to_line = junction_xs["to_line"].iloc[0]
        to_line = first_to_line
        while True:
            xs = junction_xs[junction_xs["river_reach_rs"] == to_line]
            coords += list(xs.iloc[0].geometry.coords)
            to_line = xs["to_line"].iloc[0]
            if to_line == first_to_line:
                break
        return Polygon(coords)

    def determine_junction_xs(self, xs_gdf: gpd.GeoDataFrame, junction: gpd.GeoSeries) -> gpd.GeoDataFrame:
        """Determine the cross sections that bound a junction."""
        junction_xs = []
        for us_river, us_reach in zip(junction.us_rivers.split(","), junction.us_reaches.split(",")):
            xs_us_river_reach = xs_gdf[(xs_gdf["river"] == us_river) & (xs_gdf["reach"] == us_reach)]
            junction_xs.append(
                xs_us_river_reach[xs_us_river_reach["river_station"] == xs_us_river_reach["river_station"].min()]
            )
        for ds_river, ds_reach in zip(junction.ds_rivers.split(","), junction.ds_reaches.split(",")):
            xs_ds_river_reach = xs_gdf[(xs_gdf["river"] == ds_river) & (xs_gdf["reach"] == ds_reach)].copy()
            xs_ds_river_reach["geometry"] = xs_ds_river_reach.reverse()
            junction_xs.append(
                xs_ds_river_reach[xs_ds_river_reach["river_station"] == xs_ds_river_reach["river_station"].max()]
            )
        return pd.concat(junction_xs).copy()

    def determine_xs_order(self, row: gpd.GeoSeries, junction_xs: gpd.gpd.GeoDataFrame):
        """Detemine what order cross sections bounding a junction should be in to produce a valid polygon."""
        candidate_lines = junction_xs[junction_xs["river_reach_rs"] != row["river_reach_rs"]]
        candidate_lines["distance"] = candidate_lines["start"].distance(row.end)
        return candidate_lines.loc[
            candidate_lines["distance"] == candidate_lines["distance"].min(),
            "river_reach_rs",
        ].iloc[0]

    def get_subtype_gdf(self, subtype: str) -> gpd.GeoDataFrame:
        """Get a geodataframe of a specific subtype of geometry asset."""
        tmp_objs: dict[str] = getattr(self, subtype)
        return gpd.GeoDataFrame(
            pd.concat([obj.gdf for obj in tmp_objs.values()], ignore_index=True)
        )  # TODO: may need to add some logic here for empty dicts

    def iter_labeled_gdfs(self) -> Iterator[tuple[str, gpd.GeoDataFrame]]:
        for property in self.PROPERTIES_WITH_GDF:
            gdf = self.get_subtype_gdf(property)
            yield property, gdf

    def to_gpkg(self, gpkg_path: str) -> None:
        """Write the HEC-RAS Geometry file to geopackage."""
        for subtype, gdf in self.iter_labeled_gdfs():
            gdf.to_file(gpkg_path, driver="GPKG", layer=subtype, ignore_index=True)


class SteadyFlowFile:
    """HEC-RAS Steady Flow file data."""

    def __init__(self, fpath):
        self.fpath = fpath
        with open(fpath, "r") as f:
            self.file_lines = f.readlines()

    @property
    def flow_title(self) -> str:
        return search_contents(self.file_lines, "Flow Title")

    @property
    def n_profiles(self) -> int:
        return int(search_contents(self.file_lines, "Number of Profiles"))


class UnsteadyFlowFile:
    """HEC-RAS Unsteady Flow file data."""

    def __init__(self, fpath):
        self.fpath = fpath
        with open(fpath, "r") as f:
            self.file_lines = f.readlines()

    @property
    def flow_title(self) -> str:
        return search_contents(self.file_lines, "Flow Title")

    @property
    def boundary_locations(self) -> list:
        """
        example file line:
        Boundary Location=                ,                ,        ,        ,                ,Perimeter 1     ,                ,PugetSound_Ocean_Boundary       ,
        """
        boundary_dict = []
        matches = search_contents(self.file_lines, "Boundary Location", expect_one=False)
        for line in matches:
            parts = line.split(",")
            if len(parts) >= 7:
                flow_area = parts[5].strip()
                bc_line = parts[7].strip()
                boundary_dict.append({flow_area: bc_line})
        logging.debug(f"boundary_dict:{boundary_dict}")
        return boundary_dict

    @property
    def reference_lines(self):
        return search_contents(
            self.file_lines, "Observed Rating Curve=Name=Ref Line", token=":", expect_one=False, require_one=False
        )


class QuasiUnsteadyFlowFile:
    """HEC-RAS Quasi-Unsteady Flow file data."""

    # TODO: implement this class
    pass
    # @property
    # def flow_title(self) -> str:
    #     tree = ET.parse(self.href)
    #     file_info = tree.find("FileInfo")
    #     return file_info.attrib.get("Title")


class RASHDFFile:
    """Base class for HDF assets (Plan and Geometry HDF files)."""

    def __init__(self, fpath, hdf_constructor):
        self.fpath = fpath

        self.hdf_object = hdf_constructor(fpath)
        self._root_attrs: dict | None = None
        self._geom_attrs: dict | None = None
        self._structures_attrs: dict | None = None
        self._2d_flow_attrs: dict | None = None

    @property
    def file_version(self) -> str | None:
        if self._root_attrs == None:
            self._root_attrs = self.hdf_object.get_root_attrs()
        return self._root_attrs.get("File Version")

    @property
    def units_system(self) -> str | None:
        if self._root_attrs == None:
            self._root_attrs = self.hdf_object.get_root_attrs()
        return self._root_attrs.get("Units System")

    @property
    def geometry_time(self) -> datetime.datetime | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Geometry Time").isoformat()

    @property
    def landcover_date_last_modified(self) -> datetime.datetime | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Land Cover Date Last Modified")

    @property
    def landcover_filename(self) -> str | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Land Cover Filename")

    @property
    def landcover_layername(self) -> str | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Land Cover Layername")

    @property
    def rasmapperlibdll_date(self) -> datetime.datetime | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("RasMapperLib.dll Date").isoformat()

    @property
    def si_units(self) -> bool | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("SI Units")

    @property
    def terrain_file_date(self) -> datetime.datetime | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Terrain File Date").isoformat()

    @property
    def terrain_filename(self) -> str | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Terrain Filename")

    @property
    def terrain_layername(self) -> str | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Terrain Layername")

    @property
    def geometry_version(self) -> str | None:
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_geom_attrs()
        return self._geom_attrs.get("Version")

    @property
    def bridges_culverts(self) -> int | None:
        if self._structures_attrs == None:
            self._structures_attrs = self.hdf_object.get_geom_structures_attrs()
        return self._structures_attrs.get("Bridge/Culvert Count")

    @property
    def connections(self) -> int | None:
        if self._structures_attrs == None:
            self._structures_attrs = self.hdf_object.get_geom_structures_attrs()
        return self._structures_attrs.get("Connection Count")

    @property
    def inline_structures(self) -> int | None:
        if self._structures_attrs == None:
            self._structures_attrs = self.hdf_object.get_geom_structures_attrs()
        return self._structures_attrs.get("Inline Structure Count")

    @property
    def lateral_structures(self) -> int | None:
        if self._structures_attrs == None:
            self._structures_attrs = self.hdf_object.get_geom_structures_attrs()
        return self._structures_attrs.get("Lateral Structure Count")

    @property
    def two_d_flow_cell_average_size(self) -> float | None:
        if self._2d_flow_attrs == None:
            self._2d_flow_attrs = self.hdf_object.get_geom_2d_flow_area_attrs()
        return int(np.sqrt(self._2d_flow_attrs.get("Cell Average Size")))

    @property
    def two_d_flow_cell_maximum_index(self) -> int | None:
        if self._2d_flow_attrs == None:
            self._2d_flow_attrs = self.hdf_object.get_geom_2d_flow_area_attrs()
        return self._2d_flow_attrs.get("Cell Maximum Index")

    @property
    def two_d_flow_cell_maximum_size(self) -> int | None:
        if self._2d_flow_attrs == None:
            self._2d_flow_attrs = self.hdf_object.get_geom_2d_flow_area_attrs()
        return int(np.sqrt(self._2d_flow_attrs.get("Cell Maximum Size")))

    @property
    def two_d_flow_cell_minimum_size(self) -> int | None:
        if self._2d_flow_attrs == None:
            self._2d_flow_attrs = self.hdf_object.get_geom_2d_flow_area_attrs()
        return int(np.sqrt(self._2d_flow_attrs.get("Cell Minimum Size")))

    def mesh_areas(self, crs=None, return_gdf=False) -> gpd.GeoDataFrame | Polygon | MultiPolygon:

        mesh_areas = self.hdf_object.mesh_cell_polygons()
        if mesh_areas is None or mesh_areas.empty:
            raise ValueError("No mesh areas found.")

        if mesh_areas.crs is None and crs is not None:
            mesh_areas = mesh_areas.set_crs(crs)

        if return_gdf:
            return mesh_areas
        else:
            geometries = mesh_areas["geometry"]
            return unary_union(geometries)

    @property
    def breaklines(self) -> gpd.GeoDataFrame | None:
        breaklines = self.hdf_object.breaklines()

        if breaklines is None or breaklines.empty:
            raise ValueError("No breaklines found.")
        else:
            return breaklines

    @property
    def bc_lines(self) -> gpd.GeoDataFrame | None:
        bc_lines = self.hdf_object.bc_lines()

        if bc_lines is None or bc_lines.empty:
            raise ValueError("No boundary condition lines found.")
        else:
            return bc_lines

    @property
    def landcover_filename(self) -> str | None:
        # broken example property which would give a filename to use when linking assets together
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_attrs("geom_or_something")
        return self._geom_attrs.get("land_cover_filename")

    def associate_related_assets(self, asset_dict: dict[str, Asset]) -> None:
        if self.landcover_filename:
            landcover_asset = asset_dict[self.parent.joinpath(self.landcover_filename).resolve()]
            self.extra_fields["ras:landcover_file"] = landcover_asset.href


class PlanHDFFile(RASHDFFile):

    def __init__(self, fpath: str, **kwargs):
        super().__init__(fpath, RasPlanHdf, **kwargs)

        self.hdf_object = RasPlanHdf(fpath)
        self._plan_info_attrs = None
        self._plan_parameters_attrs = None
        self._meteorology_attrs = None

    @property
    def plan_information_base_output_interval(self) -> str | None:
        # example property to show pattern: if attributes in which property is found is not loaded, load them
        # then use key for the property in the dictionary of attributes to retrieve the property
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Base Output Interval")

    @property
    def plan_information_computation_time_step_base(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Computation Time Step Base")

    @property
    def plan_information_flow_filename(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Flow Filename")

    @property
    def plan_information_geometry_filename(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Geometry Filename")

    @property
    def plan_information_plan_filename(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Plan Filename")

    @property
    def plan_information_plan_name(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Plan Name")

    @property
    def plan_information_project_filename(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Project Filename")

    @property
    def plan_information_project_title(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Project Title")

    @property
    def plan_information_simulation_end_time(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Simulation End Time").isoformat()

    @property
    def plan_information_simulation_start_time(self):
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("Simulation Start Time").isoformat()

    @property
    def plan_parameters_1d_flow_tolerance(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Flow Tolerance")

    @property
    def plan_parameters_1d_maximum_iterations(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Maximum Iterations")

    @property
    def plan_parameters_1d_maximum_iterations_without_improvement(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Maximum Iterations Without Improvement")

    @property
    def plan_parameters_1d_maximum_water_surface_error_to_abort(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Maximum Water Surface Error To Abort")

    @property
    def plan_parameters_1d_storage_area_elevation_tolerance(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Storage Area Elevation Tolerance")

    @property
    def plan_parameters_1d_theta(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Theta")

    @property
    def plan_parameters_1d_theta_warmup(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Theta Warmup")

    @property
    def plan_parameters_1d_water_surface_elevation_tolerance(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D Water Surface Elevation Tolerance")

    @property
    def plan_parameters_1d2d_gate_flow_submergence_decay_exponent(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D Gate Flow Submergence Decay Exponent")

    @property
    def plan_parameters_1d2d_is_stablity_factor(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D IS Stablity Factor")

    @property
    def plan_parameters_1d2d_ls_stablity_factor(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D LS Stablity Factor")

    @property
    def plan_parameters_1d2d_maximum_number_of_time_slices(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D Maximum Number of Time Slices")

    @property
    def plan_parameters_1d2d_minimum_time_step_for_slicinghours(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D Minimum Time Step for Slicing(hours)")

    @property
    def plan_parameters_1d2d_number_of_warmup_steps(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D Number of Warmup Steps")

    @property
    def plan_parameters_1d2d_warmup_time_step_hours(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D Warmup Time Step (hours)")

    @property
    def plan_parameters_1d2d_weir_flow_submergence_decay_exponent(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D-2D Weir Flow Submergence Decay Exponent")

    @property
    def plan_parameters_1d2d_maxiter(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("1D2D MaxIter")

    @property
    def plan_parameters_2d_equation_set(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("2D Equation Set")

    @property
    def plan_parameters_2d_names(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("2D Names")

    @property
    def plan_parameters_2d_volume_tolerance(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("2D Volume Tolerance")

    @property
    def plan_parameters_2d_water_surface_tolerance(self):
        if self._plan_parameters_attrs == None:
            self._plan_parameters_attrs = self.hdf_object.get_plan_param_attrs()
        return self._plan_parameters_attrs.get("2D Water Surface Tolerance")

    @property
    def meteorology_dss_filename(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("DSS Filename")

    @property
    def meteorology_dss_pathname(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("DSS Pathname")

    @property
    def meteorology_data_type(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("Data Type")

    @property
    def meteorology_mode(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("Mode")

    @property
    def meteorology_raster_cellsize(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("Raster Cellsize")

    @property
    def meteorology_source(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("Source")

    @property
    def meteorology_units(self):
        if self._meteorology_attrs == None:
            self._meteorology_attrs = self.hdf_object.get_meteorology_precip_attrs()
        return self._meteorology_attrs.get("Units")


class GeometryHDFFile(RASHDFFile):

    def __init__(self, fpath: str, crs=None, **kwargs):
        super().__init__(fpath, RasGeomHdf, **kwargs)

        self.hdf_object = RasGeomHdf(fpath)
        self.crs = crs
        logging.info("creating pyproj crs")
        if self.crs is None:
            try:
                self.crs = CRS.from_user_input(self.hdf_object.projection())
                logging.debug(f"Projection found in file {fpath}: {self.crs}")
            except CRSError:
                raise ValueError(f"CRS was not given and could not be extracted from from {fpath}")

        logging.info("done getting crs")
        self._plan_info_attrs = None
        self._plan_parameters_attrs = None
        self._meteorology_attrs = None

    @property
    def cross_sections(self) -> int | None:
        pass

    @property
    def reference_lines(self) -> gpd.GeoDataFrame | None:

        ref_lines = self.hdf_object.reference_lines()

        if ref_lines is None or ref_lines.empty:
            logging.warning("No reference lines found.")
        else:
            return ref_lines
