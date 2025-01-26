import logging
import math
from enum import Enum
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import LineString, Point

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
        return search_contents(self.file_lines, "Model Description", token=":")

    @property
    def project_status(self) -> str:
        return search_contents(self.file_lines, "Status of Model", token=":")

    @property
    def project_units(self) -> str | None:
        for line in self.file_lines:
            if "Units" in line:
                return " ".join(line.split(" ")[:-1])

    @property
    def plan_current(self) -> str | None:
        try:
            suffix = search_contents(self.file_lines, "Current Plan", expect_one=True)
            return self.name_from_suffix(suffix)
        except Exception:
            logging.warning("Ras model has no current plan")
            return None

    @property
    def ras_version(self) -> str | None:
        version = search_contents(self.file_lines, "Program Version", token="=", expect_one=False)
        if version == []:
            version = search_contents(self.file_lines, "Program and Version", token=":", expect_one=False)
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
        suffixes = search_contents(self.file_lines, "Flow File", expect_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]

    @property
    def quasi_unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "QuasiSteady File", expect_one=False)
        return [name_from_suffix(self.fpath, i) for i in suffixes]

    @property
    def unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Unsteady File", expect_one=False)
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
        matches = search_contents(self.file_lines, "Breach Loc", expect_one=False)
        for line in matches:
            parts = line.split(",")
            if len(parts) >= 4:
                key = parts[4].strip()
                breach_dict[key] = eval(parts[3].strip())
        logging.info(breach_dict)
        return breach_dict
