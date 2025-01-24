import datetime
import json
import logging
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import contextily as ctx
import numpy as np
from typing import Any, Callable, Iterator, TypeAlias
from shapely.ops import unary_union
import geopandas as gpd
import jsonschema
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from pystac import Asset, MediaType
from pystac.extensions.projection import ProjectionExtension
from rashdf import RasGeomHdf, RasHdf, RasPlanHdf
from shapely import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    make_valid,
    to_geojson,
    union_all,
)

from .errors import GeometryAssetInvalidCRSError
from .ras_utils import (
    data_pairs_from_text_block,
    delimited_pairs_to_lists,
    search_contents,
    text_block_from_start_end_str,
    text_block_from_start_str_length,
    text_block_from_start_str_to_empty_line,
)
from .text_utils import River, XS, Structure, Reach, Junction, StorageArea, Connection

RAS_EXTENSION_PATH = os.path.join(os.path.dirname(__file__), "extension/schema.json")
with open(RAS_EXTENSION_PATH, "r") as f:
    data = json.load(f)
RAS_EXTENSION_DICT: dict[str, Any] = data


def extract_schema_definition(definition_name: str) -> dict[str, Any]:
    """Extract asset specific schema from ras extension schema"""

    definitions: dict = deepcopy(RAS_EXTENSION_DICT["definitions"])
    ras_schema = definitions[definition_name]
    schema_specific_definitions = {}
    for internal_definition_link in collect_definition_links(ras_schema):
        internal_definition_name = os.path.basename(internal_definition_link)
        definition_value = definitions[internal_definition_name]
        schema_specific_definitions[internal_definition_name] = definition_value
    if len(schema_specific_definitions) > 0:
        ras_schema["definitions"] = schema_specific_definitions
    with open(f"{definition_name}_schema.json", "w") as f:
        json.dump(ras_schema, f)
    return ras_schema


def collect_definition_links(schema: dict[str, Any]) -> Iterator[str]:
    for k, v in schema.items():
        if k == "$ref":
            if "#/definitions/" in v:
                yield v
            elif "#" in v:
                raise ValueError(f"internal link found in key value pair {k}: {v} which is not found in #/definitions")
        elif isinstance(v, dict):
            yield from collect_definition_links(v)
        elif isinstance(v, list):
            for list_entry in v:
                if isinstance(list_entry, dict):
                    yield from collect_definition_links(list_entry)


class GenericAsset(Asset):
    """A class to handle text-based assets."""
    def __init__(
        self,
        href: str,
        title: str | None,
        description: str | None,
        media_type: str | MediaType | None,
        roles: list[str] | None,
        extra_fields: dict[str, Any] | None,
    ):
        roles = list(set(roles))
        href_path = Path(href)
        self.parent = href_path.parent
        self.name = href_path.name
        self.stem = href_path.stem
        if title == None:
            title = self.name
        # Avoid reading binary files like PNGs
        if not href.lower().endswith((".png", ".jpg", ".jpeg")):
            with open(href) as f:
                self.file_lines = f.read().splitlines()
        super().__init__(href, title, description, media_type, roles, extra_fields)

    def name_from_suffix(self, suffix: str) -> str:
        return self.stem + "." + suffix

    @property
    def program_version(self) -> str:
        """The HEC-RAS version last used to modify the file."""
        return search_contents(self.file_lines, "Program Version", expect_one=False)

    @property
    def short_summary(self):
        return {"title": self.ras_title, "file": str(self.name)}


class ProjectAsset(GenericAsset):

    def __init__(self, project_file: str, **kwargs):
        media_type = MediaType.TEXT
        roles: list[str] = kwargs.get("roles", [])
        roles.extend(["project-file", "ras-file"])
        super().__init__(
            project_file,
            kwargs.get("title"),
            kwargs.get("description", "The HEC-RAS project file."),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.ras_schema = extract_schema_definition("project")

    def populate(self) -> None:
        """Populate project asset properties and get rid of requirements for properties which are defined after other assets are associated with this asset
        (plan_current, plan_files, geometry_files, steady_flow_files, quasi_unsteady_flow_files, and unsteady_flow_files)"""

        pre_asset_association_schema = deepcopy(self.ras_schema)
        required_property_names: list[str] = pre_asset_association_schema["required"]
        for asset_associated_property in [
            "ras:plan_current",
            "ras:plan_files",
            "ras:geometry_files",
            "ras:steady_flow_files",
            "ras:quasi_unsteady_flow_files",
            "ras:unsteady_flow_files",
        ]:
            required_property_names.remove(asset_associated_property)
        pre_asset_association_schema["required"] = required_property_names
        self.extra_fields["ras:project_title"] = self.project_title
        self.extra_fields["ras:project_units"] = self.project_units
        self.extra_fields["ras:ras_version"] = self.ras_version
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, pre_asset_association_schema, jsonschema.Draft7Validator)

    @property
    @lru_cache
    def project_title(self) -> str:
        title = search_contents(self.file_lines, "Proj Title")
        return title

    @property
    @lru_cache
    def project_units(self) -> str | None:
        for line in self.file_lines:
            if "Units" in line:
                units = " ".join(line.split(" ")[:-1])
                self.extra_fields["ras:project_units"] = units
                return units

    @property
    @lru_cache
    def plan_current(self) -> str | None:
        try:
            suffix = search_contents(self.file_lines, "Current Plan", expect_one=True)
            return self.name_from_suffix(suffix)
        except Exception:
            logging.warning("Ras model has no current plan")
            return None

    @property
    def ras_version(self) -> str | None:
        try:
            return search_contents(self.file_lines, "Program Version", expect_one=True)
        except ValueError:
            return None

    @property
    @lru_cache
    def plan_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Plan File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def geometry_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Geom File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def steady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Flow File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def quasi_unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "QuasiSteady File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_lines, "Unsteady File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    def associate_related_plans(self, asset_dict: dict[str, Asset]) -> None:
        plan_file_list: list[str] = []
        for plan_file in self.plan_files:
            asset = asset_dict[plan_file]
            if plan_file == self.plan_current:
                self.extra_fields["ras:plan_current"] = asset.href
            plan_file_list.append(asset.href)
        self.extra_fields["ras:plan_files"] = plan_file_list

    def associate_related_geometries(self, asset_dict: dict[str, Asset]) -> None:
        geom_file_list: list[str] = []
        for geom_file in self.geometry_files:
            asset = asset_dict[geom_file]
            geom_file_list.append(asset.href)
        self.extra_fields["ras:geometry_files"] = geom_file_list

    def associate_related_steady_flows(self, asset_dict: dict[str, Asset]) -> None:
        steady_flow_file_list: list[str] = []
        for steady_flow_file in self.steady_flow_files:
            asset = asset_dict[steady_flow_file]
            steady_flow_file_list.append(asset.href)
        self.extra_fields["ras:steady_flow_files"] = steady_flow_file_list

    def associate_related_quasi_unsteady_flows(self, asset_dict: dict[str, Asset]) -> None:
        quasi_unsteady_flow_file_list: list[str] = []
        for quasi_unsteady_flow_file in self.quasi_unsteady_flow_files:
            asset = asset_dict[quasi_unsteady_flow_file]
            quasi_unsteady_flow_file_list.append(asset.href)
        self.extra_fields["ras:quasi_unsteady_flow_files"] = quasi_unsteady_flow_file_list

    def associate_related_unsteady_flows(self, asset_dict: dict[str, Asset]) -> None:
        unsteady_flow_file_list: list[str] = []
        for unsteady_file in self.unsteady_flow_files:
            asset = asset_dict[unsteady_file]
            unsteady_flow_file_list.append(asset.href)
        self.extra_fields["ras:unsteady_flow_files"] = unsteady_flow_file_list

    def associate_related_assets(self, asset_dict: dict[str, Asset]) -> None:
        self.associate_related_plans(asset_dict)
        self.associate_related_geometries(asset_dict)
        self.associate_related_steady_flows(asset_dict)
        self.associate_related_quasi_unsteady_flows(asset_dict)
        self.associate_related_unsteady_flows(asset_dict)
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, self.ras_schema, jsonschema.Draft7Validator)


class PlanAsset(GenericAsset):

    def __init__(self, plan_file: str, **kwargs):
        media_type = MediaType.TEXT
        roles: list[str] = kwargs.get("roles", [])
        roles.extend(["plan-file", "ras-file"])
        super().__init__(
            plan_file,
            kwargs.get("title"),
            kwargs.get(
                "description",
                "The plan file which contains a list of associated input files and all simulation options.",
            ),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.ras_schema = extract_schema_definition("plan")

    def populate(self) -> None:
        # get rid of requirements for properties which are defined after other assets are associated with this asset (geometry_file, one of [steady_flow_file, quasi_unsteady_flow_file, unsteady_flow_file])
        pre_asset_association_schema = self.ras_schema
        required_property_names: list[str] = pre_asset_association_schema["required"]
        for asset_associated_property in ["ras:geometry_file"]:
            required_property_names.remove(asset_associated_property)
        pre_asset_association_schema["required"] = required_property_names
        del pre_asset_association_schema["oneOf"]
        self.extra_fields["ras:plan_title"] = self.plan_title
        self.extra_fields["ras:short_identifier"] = self.short_identifier
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, pre_asset_association_schema, jsonschema.Draft7Validator)

    @property
    def plan_title(self) -> str:
        # gets title and adds to asset properties
        title = search_contents(self.file_lines, "Plan Title")
        self.extra_fields["ras:plan_title"] = title

    @property
    def geometry_file(self) -> str:
        suffix = search_contents(self.file_lines, "Geom File", expect_one=True)
        return self.name_from_suffix(suffix)

    @property
    def flow_file(self) -> str:
        suffix = search_contents(self.file_lines, "Flow File", expect_one=True)
        return self.name_from_suffix(suffix)

    @property
    def short_identifier(self) -> str:
        return search_contents(self.file_lines, "Short Identifier", expect_one=True)

    def associate_related_assets(self, asset_dict: dict[str, Asset]) -> None:
        primary_geometry_asset = asset_dict[self.geometry_file]
        self.extra_fields["ras:geometry_file"] = primary_geometry_asset.href
        primary_flow_asset = asset_dict[self.flow_file]
        if isinstance(primary_flow_asset, SteadyFlowAsset):
            property_name = "ras:steady_flow_file"
        elif isinstance(primary_flow_asset, QuasiUnsteadyFlowAsset):
            property_name = "ras:quasi_unsteady_flow_file"
        elif isinstance(primary_flow_asset, UnsteadyFlowAsset):
            property_name = "ras:unsteady_flow_file"
        else:
            logging.warning(
                f"Asset being linked to plan asset {self.plan_title} is not one of the following: ['SteadyFlowAsset', 'QuasiUnsteadyFlowAsset', 'UnsteadyFlowAsset']; cannot provide a link type for the link object being created"
            )
        self.extra_fields[property_name] = primary_flow_asset.href
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, self.ras_schema, jsonschema.Draft7Validator)


class GeometryAsset(GenericAsset):
    PROPERTIES_WITH_GDF = ["reaches", "junctions", "cross_sections", "structures"]

    def __init__(self, geom_file: str, crs: str, **kwargs):
        self.pyproj_crs = self.validate_crs(crs)
        media_type = MediaType.TEXT
        roles: list[str] = kwargs.get("roles", [])
        roles.extend(["geometry-file", "ras-file"])
        super().__init__(
            geom_file,
            kwargs.get("title"),
            kwargs.get(
                "description",
                "The geometry file which contains cross-sectional, hydraulic structures, and modeling approach data.",
            ),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.crs = crs
        self.ras_schema = extract_schema_definition("geometry")

    @staticmethod
    def validate_crs(crs: str) -> CRS:
        try:
            return CRS.from_user_input(crs)
        except Exception as exc:
            raise GeometryAssetInvalidCRSError(*exc.args)

    def populate(self) -> None:
        self.extra_fields["ras:geometry_title"] = self.geom_title
        self.extra_fields["ras:rivers"] = len(self.rivers)
        self.extra_fields["ras:reaches"] = len(self.reaches)
        self.extra_fields["ras:cross_sections"] = {
            "total": len(self.cross_sections),
            "user_input_xss": len([xs for xs in self.cross_sections.values() if not xs.is_interpolated]),
            "interpolated": len([xs for xs in self.cross_sections.values() if xs.is_interpolated]),
        }
        self.extra_fields["ras:culverts"] = len(
            [s for s in self.structures.values() if s.type == StructureType.CULVERT]
        )
        self.extra_fields["ras:bridges"] = len([s for s in self.structures.values() if s.type == StructureType.BRIDGE])
        self.extra_fields["ras:multiple_openings"] = len(
            [s for s in self.structures.values() if s.type == StructureType.MULTIPLE_OPENING]
        )
        self.extra_fields["ras:inline_structures"] = len(
            [s for s in self.structures.values() if s.type == StructureType.INLINE_STRUCTURE]
        )
        self.extra_fields["ras:lateral_structures"] = len(
            [s for s in self.structures.values() if s.type == StructureType.LATERAL_STRUCTURE]
        )
        # TODO: implement actual logic for populating storage areas, 2d flow areas, and sa connections
        self.extra_fields["ras:storage_areas"] = 0
        self.extra_fields["ras:2d_flow_areas"] = {
            "2d_flow_areas": 0,
            "total_cells": 0,
        }
        self.extra_fields["ras:sa_connections"] = 0
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, self.ras_schema, jsonschema.Draft7Validator)

        if len(self.cross_sections) > 0:
            proj = ProjectionExtension.ext(self)
            proj.geometry = json.loads(to_geojson(self.concave_hull))
            proj.bbox = self.concave_hull.bounds
            proj.wkt2 = self.pyproj_crs.to_wkt()

    @property
    def geom_title(self) -> str:
        return search_contents(self.file_lines, "Geom Title")

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
        river_reaches = search_contents(self.file_lines, "River Reach", expect_one=False)
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
    @lru_cache
    def concave_hull(self) -> Polygon:
        """Compute and return the concave hull (polygon) for cross sections."""
        polygons = []
        xs_gdf = pd.concat([xs.gdf for xs in self.cross_sections.values()], ignore_index=True)
        for river_reach in xs_gdf["river_reach"].unique():
            xs_subset: gpd.GeoSeries = xs_gdf[xs_gdf["river_reach"] == river_reach]
            points = xs_subset.boundary.explode(index_parts=True).unstack()
            points_last_xs = [Point(coord) for coord in xs_subset["geometry"].iloc[-1].coords]
            points_first_xs = [Point(coord) for coord in xs_subset["geometry"].iloc[0].coords[::-1]]
            polygon = Polygon(points_first_xs + list(points[0]) + points_last_xs + list(points[1])[::-1])
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
        tmp_objs: dict[str, RasGeometryClass] = getattr(self, subtype)
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


class SteadyFlowAsset(GenericAsset):

    def __init__(self, steady_flow_file: str, **kwargs):
        media_type = MediaType.TEXT
        roles: list[str] = kwargs.get("roles", [])
        roles.extend(["steady-flow-file", "ras-file"])
        super().__init__(
            steady_flow_file,
            kwargs.get("title"),
            kwargs.get(
                "description",
                "Steady Flow file which contains profile information, flow data, and boundary conditions.",
            ),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.ras_schema = extract_schema_definition("steady_flow")

    def populate(self) -> None:
        self.extra_fields["ras:flow_title"] = self.flow_title
        self.extra_fields["ras:number_of_profiles"] = self.n_profiles
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, self.ras_schema, jsonschema.Draft7Validator)

    @property
    def flow_title(self) -> str:
        return search_contents(self.file_lines, "Flow Title")

    @property
    def n_profiles(self) -> int:
        return int(search_contents(self.file_lines, "Number of Profiles"))


class QuasiUnsteadyFlowAsset(GenericAsset):

    def __init__(self, quasi_unsteady_flow_file: str, **kwargs):
        media_type = MediaType.TEXT
        roles: list[str] = kwargs.get("roles", [])
        roles.extend(["quasi-unsteady-flow-file", "ras-file"])
        super().__init__(
            quasi_unsteady_flow_file,
            kwargs.get("title"),
            kwargs.get("description", "Quasi-Unsteady Flow file."),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.ras_schema = extract_schema_definition("quasi_unsteady_flow")

    def populate(self) -> None:
        self.extra_fields["ras:flow_title"] = self.flow_title
        as_dict = self.to_dict()
        jsonschema.validate(as_dict, self.ras_schema, jsonschema.Draft7Validator)

    @property
    def flow_title(self) -> str:
        tree = ET.parse(self.href)
        file_info = tree.find("FileInfo")
        return file_info.attrib.get("Title")


class UnsteadyFlowAsset(GenericAsset):

    def __init__(self, unsteady_flow_file: str, **kwargs):
        media_type = MediaType.TEXT
        roles: list[str] = kwargs.get("roles", [])
        roles.extend(["unsteady-flow-file", "ras-file"])
        super().__init__(
            unsteady_flow_file,
            kwargs.get("title"),
            kwargs.get(
                "description",
                "The unsteady file contains hydrographs amd initial conditions, as well as any flow options.",
            ),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.ras_schema = extract_schema_definition("unsteady_flow")

    def populate(self) -> None:
        self.extra_fields["ras:flow_title"] = self.flow_title
        # as_dict = self.to_dict()
        # jsonschema.validate(as_dict, self.ras_schema, jsonschema.Draft7Validator)

    @property
    def flow_title(self) -> str:
        return search_contents(self.file_lines, "Flow Title")


class HdfAsset(Asset):
    # class to represent stac asset with properties shared between plan hdf and geom hdf
    def __init__(
        self,
        hdf_file: str,
        hdf_constructor: Callable[[str], RasHdf],
        title: str | None = None,
        description: str | None = None,
        roles: list[str] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ):
        hdf_file_path = Path(hdf_file)
        self.parent = hdf_file_path.parent
        self.name = hdf_file_path.name
        if title == None:
            title = self.name
        media_type = MediaType.HDF
        if roles == None:
            roles = []
        roles.append("ras-file")
        super().__init__(hdf_file, title, description, media_type, roles, extra_fields)
        self.hdf_object = hdf_constructor(hdf_file)
        self._root_attrs: dict | None = None
        self._geom_attrs: dict | None = None
        self._structures_attrs: dict | None = None
        self._2d_flow_attrs: dict | None = None

    def populate(
        self,
        optional_property_dict: dict[str, str],
        required_property_dict: dict[str, str],
    ) -> None:
        # go through dictionary of stac property names and class property names, only adding property to extra fields if the value is not None
        for stac_property_name, class_property_name in optional_property_dict.items():
            property_value = getattr(self, class_property_name)
            if property_value != None:
                self.extra_fields[stac_property_name] = property_value
        # go through dictionary of stac property names and class property names, adding all properties to extra fields regardless of value
        for stac_property_name, class_property_name in required_property_dict.items():
            property_value = getattr(self, class_property_name)
            self.extra_fields[stac_property_name] = property_value

    @property
    def file_version(self) -> str | None:
        # example property to show pattern: if attributes in which property is found is not loaded, load them
        # then use key for the property in the dictionary of attributes to retrieve the property
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

    @lru_cache
    def mesh_areas(self, crs, return_gdf=False) -> gpd.GeoDataFrame | Polygon | MultiPolygon:

        mesh_areas = self.hdf_object.mesh_cell_polygons()
        if mesh_areas is None or mesh_areas.empty:
            raise ValueError("No mesh areas found.")

        if mesh_areas.crs and mesh_areas.crs != crs:
            mesh_areas = mesh_areas.to_crs(crs)

        if return_gdf:
            return mesh_areas
        else:
            geometries = mesh_areas["geometry"]
            return unary_union(geometries)

    @property
    @lru_cache
    def breaklines(self) -> gpd.GeoDataFrame | None:
        breaklines = self.hdf_object.breaklines()

        if breaklines is None or breaklines.empty:
            raise ValueError("No breaklines found.")
        else:
            return breaklines

    @property
    @lru_cache
    def bc_lines(self) -> gpd.GeoDataFrame | None:
        bc_lines = self.hdf_object.bc_lines()

        if bc_lines is None or bc_lines.empty:
            raise ValueError("No boundary condition lines found.")
        else:
            return bc_lines

    @property
    @lru_cache
    def landcover_filename(self) -> str | None:
        # broken example property which would give a filename to use when linking assets together
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_attrs("geom_or_something")
        return self._geom_attrs.get("land_cover_filename")

    def associate_related_assets(self, asset_dict: dict[str, Asset]) -> None:
        if self.landcover_filename:
            landcover_asset = asset_dict[self.parent.joinpath(self.landcover_filename).resolve()]
            self.extra_fields["ras:landcover_file"] = landcover_asset.href


class PlanHdfAsset(HdfAsset):
    # class to represent stac asset for plan HDF file associated with model
    def __init__(self, hdf_file: str, **kwargs):
        description = kwargs.get("description", "The HEC-RAS plan HDF file.")
        super().__init__(hdf_file, RasPlanHdf, description=description)
        self.hdf_object: RasPlanHdf
        self._plan_info_attrs = None
        self._plan_parameters_attrs = None
        self._meteorology_attrs = None

    def populate(self) -> None:
        plan_hdf_properties = {
            "plan_information:Base_Output_Interval": "plan_information_base_output_interval",
            "plan_information:Computation_Time_Step_Base": "plan_information_computation_time_step_base",
            "plan_information:Flow_Filename": "plan_information_flow_filename",
            "plan_information:Geometry_Filename": "plan_information_geometry_filename",
            "plan_information:Plan_Filename": "plan_information_plan_filename",
            "plan_information:Plan_Name": "plan_information_plan_name",
            "plan_information:Project_Filename": "plan_information_project_filename",
            "plan_information:Simulation_End_Time": "plan_information_simulation_end_time",
            "plan_information:Simulation_Start_Time": "plan_information_simulation_start_time",
            "plan_parameters:1D_Flow_Tolerance": "plan_parameters_1d_flow_tolerance",
            "plan_parameters:2D_Equation_Set": "plan_parameters_2d_equation_set",
            "meteorology:DSS_Filename": "meteorology_dss_filename",
            "meteorology:Mode": "meteorology_mode",
        }
        super().populate(plan_hdf_properties, {})

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


class GeometryHdfAsset(HdfAsset):
    # class to represent stac asset for geom HDF file associated with model
    def __init__(self, hdf_file: str, **kwargs):
        description = kwargs.get("description", "The HEC-RAS geometry HDF file.")
        super().__init__(hdf_file, RasGeomHdf.open_uri, description=description)
        self.hdf_object: RasGeomHdf
        self.hdf_file = hdf_file

    @property
    def cross_sections(self) -> int | None:
        pass

    @property
    @lru_cache
    def reference_lines(self) -> gpd.GeoDataFrame | None:

        ref_lines = self.hdf_object.reference_lines()

        if ref_lines is None or ref_lines.empty:
            raise ValueError("No reference lines found.")
        else:
            return ref_lines

    def populate(self) -> None:
        # determine optional and required properties
        geom_hdf_properties = {
            "2D_Flow_Areas:Cell_Average_Size": "two_d_flow_cell_average_size",
            "2D_Flow_Areas:Cell_Maximum_Size": "two_d_flow_cell_maximum_size",
            "2D_Flow_Areas:Cell_Minimum_Size": "two_d_flow_cell_minimum_size",
            "Structures:Bridge_Culvert_Count": "bridges_culverts",
            "Structures:Connection Count": "connections",
            "Structures:Inline_Structure_Count": "inline_structures",
            "Structures:Lateral_Structure_Count": "lateral_structures",
            "File_Version":"file_version",
            "Units_System":"units_system",
            "Geometry_Time":"geometry_time",
            "File_Version":"file_version",
            "Landcover_Filename": "landcover_filename",
            "Terrain_Filename":"terrain_filename",
            "Geometry_Version": "geometry_version"
        }
        super().populate(geom_hdf_properties, {})

    def _plot_mesh_areas(self, ax, mesh_polygons: gpd.GeoDataFrame) -> list[Line2D]:
        """
        Plots mesh areas on the given axes.
        """
        mesh_polygons.plot(
            ax=ax,
            edgecolor="silver",
            facecolor="none",
            linestyle="-",
            alpha=0.7,
            label="Mesh Polygons",
        )
        legend_handle = [
            Line2D(
                [0],
                [0],
                color="silver",
                linestyle="-",
                linewidth=2,
                label="Mesh Polygons",
            )
        ]
        return legend_handle

    def _plot_breaklines(self, ax, breaklines: gpd.GeoDataFrame) -> list[Line2D]:
        """
        Plots breaklines on the given axes.
        """
        breaklines.plot(ax=ax, edgecolor="red", linestyle="-", alpha=0.3, label="Breaklines")
        legend_handle = [
            Line2D(
                [0],
                [0],
                color="red",
                linestyle="-",
                alpha=0.4,
                linewidth=2,
                label="Breaklines",
            )
        ]
        return legend_handle

    def _plot_bc_lines(self, ax, bc_lines: gpd.GeoDataFrame) -> list[Line2D]:
        """
        Plots boundary condition lines on the given axes.
        """
        legend_handles = [
            Line2D([0], [0], color="none", linestyle="None", label="BC Lines"),
        ]
        colors = plt.cm.get_cmap("Dark2", len(bc_lines))

        for bc_line, color in zip(bc_lines.itertuples(), colors.colors):
            x_coords, y_coords = bc_line.geometry.xy
            ax.plot(
                x_coords,
                y_coords,
                color=color,
                linestyle="-",
                linewidth=2,
                label=bc_line.name,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    label=bc_line.name,
                )
            )
        return legend_handles

    def _add_thumbnail_asset(self, filepath: str) -> None:
        """Add the thumbnail image as an asset with a relative href."""

        filename = os.path.basename(filepath)

        if filepath.startswith("s3://"):
            media_type = "image/png"
        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Thumbnail file not found: {filepath}")
            media_type = "image/png"

        return GenericAsset(
            href=filename,
            title="Model Thumbnail",
            description="Thumbnail image for the model",
            media_type=media_type,
            roles=["thumbnail"],
            extra_fields=None,
        )

    # def get_primary_geom(self):
    #     # TODO: This functions should probably be more robust and check for the primary geometry file based on some criteria
    #     geom_hdf_assets = [asset for asset in self._geom_files if isinstance(asset, GeometryHdfAsset)]
    #     if len(geom_hdf_assets) == 0:
    #         raise FileNotFoundError("No 2D geometry found")
    #     elif len(geom_hdf_assets) > 1:
    #         primary_geom_hdf_asset = next(asset for asset in geom_hdf_assets if ".g01" in asset.hdf_file)
    #     else:
    #         primary_geom_hdf_asset = geom_hdf_assets[0]
    #     return primary_geom_hdf_asset

    def thumbnail(
        self,
        add_asset: bool,
        write: bool,
        layers: list,
        title: str = "Model_Thumbnail",
        add_usgs_properties: bool = False,
        crs = "EPSG:4326",
        thumbnail_dest: str =  None,
    ):
        """Create a thumbnail figure for each geometry hdf file, including
        various geospatial layers such as USGS gages, mesh areas,
        breaklines, and boundary condition (BC) lines. If `add_asset` or `write`
        is `True`, the function saves the thumbnail to a file and optionally
        adds it as an asset.

        Parameters
        ----------
        add_asset : bool
            Whether to add the thumbnail as an asset in the asset dictionary. If true then it also writes the thumbnail to a file.
        write : bool
            Whether to save the thumbnail image to a file.
        layers : list
            A list of model layers to include in the thumbnail plot.
            Options include "usgs_gages", "mesh_areas", "breaklines", and "bc_lines".
        title : str, optional
            Title of the figure, by default "Model Thumbnail".
        add_usgs_properties : bool, optional
            If usgs_gages is included in layers, adds USGS metadata to the STAC item properties. Defaults to false.
        """

        fig, ax = plt.subplots(figsize=(12, 12))
        legend_handles = []

        for layer in layers:
            try:
                # if layer == "usgs_gages":
                #     if add_usgs_properties:
                #         gages_gdf = self.get_usgs_data(True, geom_asset=geom_asset)
                #     else:
                #         gages_gdf = self.get_usgs_data(False, geom_asset=geom_asset)
                #     gages_gdf_geo = gages_gdf.to_crs(self.crs)
                #     legend_handles += self._plot_usgs_gages(ax, gages_gdf_geo)
                # else:
                #     if not hasattr(geom_asset, layer):
                #         raise AttributeError(f"Layer {layer} not found in {geom_asset.hdf_file}")

                    # if layer == "mesh_areas":
                    #     layer_data = geom_asset.mesh_areas(self.crs, return_gdf=True)
                    # else:
                    #     layer_data = getattr(geom_asset, layer)

                    # if layer_data.crs is None:
                    #     layer_data.set_crs(self.crs, inplace=True)
                    # layer_data_geo = layer_data.to_crs(self.crs)

                    if layer == "mesh_areas":
                        mesh_areas_data = self.mesh_areas(crs, return_gdf=True)
                        legend_handles += self._plot_mesh_areas(ax, mesh_areas_data)
                    elif layer == "breaklines":
                        breaklines_data = self.breaklines
                        breaklines_data_geo = breaklines_data.to_crs(crs)
                        legend_handles += self._plot_breaklines(ax, breaklines_data_geo)
                    elif layer == "bc_lines":
                        bc_lines_data = self.bc_lines
                        bc_lines_data_geo = bc_lines_data.to_crs(crs)
                        legend_handles += self._plot_bc_lines(ax, bc_lines_data_geo)
            except Exception as e:
                logging.warning(f"Warning: Failed to process layer '{layer}' for {self.hdf_file}: {e}")

        # Add OpenStreetMap basemap
        ctx.add_basemap(
            ax,
            crs=crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.4,
        )
        ax.set_title(f"{title} - {os.path.basename(self.hdf_file)}", fontsize=15)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1, 0.5))


        if add_asset or write:
            hdf_ext = os.path.basename(self.hdf_file).split(".")[-2]
            filename = f"thumbnail_{hdf_ext}.png"
            base_dir = os.path.dirname(thumbnail_dest)
            filepath = os.path.join(base_dir, filename)

            if filepath.startswith("s3://"):
                img_data = io.BytesIO()
                fig.savefig(img_data, format="png", bbox_inches="tight")
                img_data.seek(0)
                save_bytes_s3(img_data, filepath)
            else:
                os.makedirs(base_dir, exist_ok=True)
                fig.savefig(filepath, dpi=80, bbox_inches="tight")

            if add_asset:
                return self._add_thumbnail_asset(filepath)

RasAsset: TypeAlias = (
    GenericAsset
    | GeometryAsset
    | PlanAsset
    | ProjectAsset
    | QuasiUnsteadyFlowAsset
    | GeometryHdfAsset
    | PlanHdfAsset
    | SteadyFlowAsset
    | UnsteadyFlowAsset
)

RasGeometryClass: TypeAlias = Reach | Junction | XS | Structure
