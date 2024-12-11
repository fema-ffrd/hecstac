import datetime
import json
import logging
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, TypeAlias

import geopandas as gpd
import pandas as pd
from errors import GeometryAssetInvalidCRSError
from pyproj import CRS
from pystac import Asset, Link, MediaType, RelType
from pystac.extensions.projection import ProjectionExtension
from ras_utils import (
    data_pairs_from_text_block,
    delimited_pairs_to_lists,
    junction_hull,
    search_contents,
    text_block_from_start_end_str,
    text_block_from_start_str_length,
    text_block_from_start_str_to_empty_line,
)
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


class LinkableAsset(Asset):
    def _set_link(self, asset: Asset, rel: RelType | str, extra_fields: dict[str, Any] | None) -> None:
        link = Link(rel, asset, extra_fields=extra_fields)
        link.set_owner(self)

    def link_child(self, asset: Asset, extra_fields: dict[str, Any] | None = None) -> None:
        self._set_link(asset, RelType.CHILD, extra_fields)

    def link_related(self, asset: Asset, extra_fields: dict[str, Any] | None = None) -> None:
        self._set_link(asset, "related", extra_fields)


class GenericAsset(LinkableAsset):
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
        self.name = Path(href).name
        self.stem = Path(href).stem
        if title == None:
            title = self.name
        with open(href) as f:
            self.file_str = f.read().splitlines()
        super().__init__(href, title, description, media_type, roles, extra_fields)

    def name_from_suffix(self, suffix: str) -> str:
        return self.stem + "." + suffix

    @property
    def program_version(self) -> str:
        """The HEC-RAS version last used to modify the file."""
        return search_contents(self.file_str, "Program Version", expect_one=False)

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

    def populate(self) -> None:
        self.extra_fields["ras:project_title"] = self.project_title
        self.extra_fields["ras:project_units"] = self.project_units
        self.extra_fields["ras:model_name"] = self.model_name
        self.extra_fields["ras:ras_version"] = self.ras_version

    @property
    @lru_cache
    def project_title(self) -> str:
        title = search_contents(self.file_str, "Proj Title")
        return title

    @property
    @lru_cache
    def project_units(self) -> str | None:
        for line in self.file_str:
            if "Units" in line:
                units = " ".join(line.split(" ")[:-1])
                self.extra_fields["ras:project_units"] = units
                return units

    @property
    @lru_cache
    def plan_current(self) -> str | None:
        try:
            suffix = search_contents(self.file_str, "Current Plan", expect_one=True)
            return self.name_from_suffix(suffix)
        except Exception:
            return None

    @property
    def model_name(self) -> str:
        pass

    @property
    def ras_version(self) -> str:
        pass

    @property
    @lru_cache
    def plan_files(self) -> list[str]:
        suffixes = search_contents(self.file_str, "Plan File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def geometry_files(self) -> list[str]:
        suffixes = search_contents(self.file_str, "Geom File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def steady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_str, "Flow File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def quasi_unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_str, "QuasiSteady File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    @property
    @lru_cache
    def unsteady_flow_files(self) -> list[str]:
        suffixes = search_contents(self.file_str, "Unsteady File", expect_one=False)
        return [self.name_from_suffix(i) for i in suffixes]

    def create_links(self, asset_dict: dict[str, Asset]) -> None:
        for plan_file in self.plan_files:
            link_type = "plan_referenced"
            # give link extra attribute denoting that linked asset is current plan if it is current plan
            if plan_file == self.plan_current:
                link_type = "plan_current"
            asset = asset_dict[plan_file]
            self.link_related(asset, {"ras:link_type", link_type})
        for geom_file in self.geometry_files:
            asset = asset_dict[geom_file]
            self.link_related(asset, {"ras:link_type", "geometry_referenced"})
        for steady_flow_file in self.steady_flow_files:
            asset = asset_dict[steady_flow_file]
            self.link_related(asset, {"ras:link_type", "steady_flow_referenced"})
        for quasi_unsteady_flow_file in self.quasi_unsteady_flow_files:
            asset = asset_dict[quasi_unsteady_flow_file]
            self.link_related(asset, {"ras:link_type", "quasi_unsteady_flow_referenced"})
        for unsteady_file in self.unsteady_flow_files:
            asset = asset_dict[unsteady_file]
            self.link_related(asset, {"ras:link_type", "unsteady_flow_referenced"})


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

    def populate(self) -> None:
        self.extra_fields["ras:plan_title"] = self.plan_title
        self.extra_fields["ras:short_id"] = self.short_id

    @property
    def plan_title(self) -> str:
        # gets title and adds to asset properties
        title = search_contents(self.file_str, "Plan Title")
        self.extra_fields["ras:plan_title"] = title

    @property
    def primary_geometry(self) -> str:
        suffix = search_contents(self.file_str, "Geom File", expect_one=True)
        return self.name_from_suffix(suffix)

    @property
    def primary_flow(self) -> str:
        suffix = search_contents(self.file_str, "Flow File", expect_one=True)
        return self.name_from_suffix(suffix)

    @property
    def short_id(self) -> str:
        return search_contents(self.file_str, "Short Identifier")

    def create_links(self, asset_dict: dict[str, Asset]) -> None:
        primary_geometry_asset = asset_dict[self.primary_geometry]
        self.link_related(primary_geometry_asset, {"ras:link_type": "geometry_referenced"})
        primary_flow_asset = asset_dict[self.primary_flow]
        flow_link_extra_fields = None
        if isinstance(primary_flow_asset, SteadyFlowAsset):
            flow_link_extra_fields = {"ras:link_type": "steady_flow_referenced"}
        elif isinstance(primary_flow_asset, QuasiUnsteadyFlowAsset):
            flow_link_extra_fields = {"ras:link_type": "quasi_unsteady_flow_referenced"}
        elif isinstance(primary_flow_asset, UnsteadyFlowAsset):
            flow_link_extra_fields = {"ras:link_type": "unsteady_flow_referenced"}
        else:
            logging.warning(
                f"Asset being linked to plan asset {self.plan_title} is not one of the following: ['SteadyFlowAsset', 'QuasiUnsteadyFlowAsset', 'UnsteadyFlowAsset']; cannot provide a link type for the link object being created"
            )
        self.link_related(primary_flow_asset, flow_link_extra_fields)


class GeometryAsset(GenericAsset):
    PROPERTIES_WITH_GDF = ["rivers", "reaches", "junctions", "cross_sections", "structures"]

    def __init__(self, geom_file: str, crs: str, **kwargs):
        self.validate_crs(crs)
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

    @staticmethod
    def validate_crs(crs: str) -> None:
        try:
            CRS.from_user_input(crs)
        except Exception as exc:
            raise GeometryAssetInvalidCRSError(*exc.args)

    def populate(self) -> None:
        self.extra_fields["ras:geom_title"] = self.geom_title
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
        self.extra_fields["ras:storage_areas"] = 0
        self.extra_fields["ras:2d_flow_areas"] = {
            "2d_flow_areas": 0,
            "total_cells": 0,
        }

        self.extra_fields["ras:sa_connections"] = 0
        proj = ProjectionExtension.ext(self, True)
        proj.geometry = json.load(to_geojson(self.concave_hull))
        proj.bbox = self.concave_hull.bounds

    @property
    def geom_title(self) -> str:
        return search_contents(self.file_str, "Geom Title")

    @property
    def rivers(self) -> dict[str, "River"]:
        """A dictionary of river_name: River (class) for the rivers contained in the HEC-RAS geometry file."""
        tmp_rivers = defaultdict(list)
        for reach in self.reaches.values():  # First, group all reaches into their respective rivers
            tmp_rivers[reach.river].append(reach.reach)
        for river, reaches in tmp_rivers.items():  # Then, create a River object for each river
            tmp_rivers[river] = River(river, reaches)
        return tmp_rivers

    @property
    def reaches(self) -> dict[str, "Reach"]:
        """A dictionary of the reaches contained in the HEC-RAS geometry file."""
        river_reaches = search_contents(self.file_str, "River Reach", expect_one=False)
        return {river_reach: Reach(self.file_str, river_reach, self.crs) for river_reach in river_reaches}

    @property
    def junctions(self) -> dict[str, "Junction"]:
        """A dictionary of the junctions contained in the HEC-RAS geometry file."""
        juncts = search_contents(self.file_str, "Junct Name", expect_one=False)
        return {junction: Junction(self.file_str, junction, self.crs) for junction in juncts}

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
        areas = search_contents(self.file_str, "Storage Area", expect_one=False)
        return {a: StorageArea(a, self.crs) for a in areas}

    @property
    def connections(self) -> dict[str, "Connection"]:
        """A dictionary of the SA/2D connections contained in the HEC-RAS geometry file."""
        connections = search_contents(self.file_str, "Connection", expect_one=False)
        return {c: Connection(c, self.crs) for c in connections}

    @property
    def datetimes(self) -> list[datetime.datetime]:
        """Get the latest node last updated entry for this geometry"""
        dts = search_contents(self.file_str, "Node Last Edited Time", expect_one=False)
        if len(dts) >= 1:
            try:
                return [datetime.datetime.strptime(d, "%b/%d/%Y %H:%M:%S") for d in dts]
            except ValueError:
                return []
        else:
            return []

    @property
    @lru_cache
    def concave_hull(self) -> Polygon:
        """Compute and return the concave hull (polygon) for cross sections."""
        polygons = []
        xs_df = pd.concat([xs.gdf for xs in self.cross_sections.values()], ignore_index=True)
        for river_reach in xs_df["river_reach"].unique():
            xs_subset: gpd.GeoSeries = xs_df[xs_df["river_reach"] == river_reach]
            points = xs_subset.boundary.explode(index_parts=True).unstack()
            points_last_xs = [Point(coord) for coord in xs_subset["geometry"].iloc[-1].coords]
            points_first_xs = [Point(coord) for coord in xs_subset["geometry"].iloc[0].coords[::-1]]
            polygon = Polygon(points_first_xs + list(points[0]) + points_last_xs + list(points[1])[::-1])
            if isinstance(polygon, MultiPolygon):
                polygons += list(polygon.geoms)
            else:
                polygons.append(polygon)
        # # Code copied but cannot be implemented because I don't know where 'xs' cross section variable is meant to have been defined
        # if len(self.junctions) > 0:
        #     for _, j in self.junction_gdf.iterrows():
        #         polygons.append(junction_hull(xs, j))
        out_hull = [union_all([make_valid(p) for p in polygons])]
        return out_hull

    def get_subtype_gdf(self, subtype: str) -> gpd.GeoDataFrame:
        """Get a geodataframe of a specific subtype of geometry asset."""
        tmp_objs: dict[str, RasGeometryClass] = getattr(self, subtype)
        return gpd.GeoDataFrame(
            pd.concat([obj.gdf for obj in tmp_objs.values()], ignore_index=True)
        )  # TODO: may need to add some logic here for empy dicts

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

    def populate(self) -> None:
        self.extra_fields["ras:flow_title"] = self.flow_title
        self.extra_fields["ras:number_of_profiles"] = self.n_profiles

    @property
    def flow_title(self) -> str:
        return search_contents(self.file_str, "Flow Title")

    @property
    def n_profiles(self) -> int:
        return int(search_contents(self.file_str, "Number of Profiles"))


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

    def populate(self) -> None:
        self.extra_fields["ras:flow_title"] = self.flow_title

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

    def populate(self) -> None:
        self.extra_fields["ras:flow_title"] = self.flow_title

    @property
    def flow_title(self) -> str:
        return search_contents(self.file_str, "Flow Title")


class HdfAsset(LinkableAsset):
    # class to represent stac asset with properties shared between plan hdf and geom hdf
    def __init__(self, hdf_file: str, hdf_constructor: Callable[[str], RasHdf], **kwargs):
        media_type = MediaType.HDF
        roles: list[str] = kwargs.get("roles", [])
        roles.append("ras-file")
        super().__init__(
            hdf_file,
            kwargs.get("title"),
            kwargs.get("description"),
            media_type,
            roles,
            kwargs.get("extra_fields"),
        )
        self.hdf_object = hdf_constructor(hdf_file)
        self._root_attrs: dict | None = None
        self._geom_attrs: dict | None = None
        self._structures_attrs: dict | None = None
        self._2d_flow_attrs: dict | None = None

    def populate(self, optional_property_dict: dict[str, str], required_property_dict: dict[str, str]) -> None:
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
        return self._root_attrs.get("version")

    @property
    def units_system(self) -> str | None:
        pass

    @property
    def geometry_time(self) -> datetime.datetime | None:
        pass

    @property
    def landcover_date_last_modified(self) -> datetime.datetime | None:
        pass

    @property
    def landcover_filename(self) -> str | None:
        pass

    @property
    def landcover_layername(self) -> str | None:
        pass

    @property
    def rasmapperlibdll_date(self) -> datetime.datetime | None:
        pass

    @property
    def si_units(self) -> bool | None:
        pass

    @property
    def terrain_file_date(self) -> datetime.datetime | None:
        pass

    @property
    def terrain_filename(self) -> str | None:
        pass

    @property
    def terrain_layername(self) -> str | None:
        pass

    @property
    def geometry_version(self) -> str | None:
        pass

    @property
    def bridges(self) -> int | None:
        pass

    @property
    def culverts(self) -> int | None:
        pass

    @property
    def connections(self) -> int | None:
        pass

    @property
    def inline_structures(self) -> int | None:
        pass

    @property
    def lateral_structures(self) -> int | None:
        pass

    @property
    def two_d_flow_cell_average_size(self) -> float | None:
        pass

    @property
    def two_d_flow_cell_maximum_index(self) -> int | None:
        pass

    @property
    def two_d_flow_cell_maximum_size(self) -> int | None:
        pass

    @property
    def two_d_flow_cell_minimum_size(self) -> int | None:
        pass

    @property
    @lru_cache
    def mesh_areas(self) -> Polygon:
        pass

    @property
    @lru_cache
    def landcover_filename(self) -> str | None:
        # broken example property which would give a filename to use when linking assets together
        if self._geom_attrs == None:
            self._geom_attrs = self.hdf_object.get_attrs("geom_or_something")
        return self._geom_attrs.get("land_cover_filename")

    def create_links(self, asset_dict: dict[str, Asset]) -> None:
        if self.landcover_filename:
            landcover_asset = asset_dict[self.landcover_filename]
            self.link_related(landcover_asset, {"ras:link_type": "landcover_referenced"})


class PlanHdfAsset(HdfAsset):
    # class to represent stac asset for plan HDF file associated with model
    def __init__(self, hdf_file: str, **kwargs):
        description = kwargs.get("description", "The HEC-RAS plan HDF file.")
        super().__init__(hdf_file, RasPlanHdf, description=description)
        self.hdf_object: RasPlanHdf
        self._plan_info_attrs = None
        self._plan_parameters_attrs = None
        self._meteorology_attrs = None

    @property
    def plan_information_base_output_interval(self) -> str | None:
        # example property to show pattern: if attributes in which property is found is not loaded, load them
        # then use key for the property in the dictionary of attributes to retrieve the property
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("base_output_interval")

    @property
    def plan_information_computation_time_step_base(self):
        pass

    @property
    def plan_information_flow_filename(self):
        pass

    @property
    def plan_information_geometry_filename(self):
        pass

    @property
    def plan_information_plan_filename(self):
        pass

    @property
    def plan_information_plan_name(self):
        pass

    @property
    def plan_information_project_filename(self):
        pass

    @property
    def plan_information_project_title(self):
        pass

    @property
    def plan_information_simulation_end_time(self):
        pass

    @property
    def plan_information_simulation_start_time(self):
        pass

    @property
    def plan_parameters_1d_flow_tolerance(self):
        pass

    @property
    def plan_parameters_1d_maximum_iterations(self):
        pass

    @property
    def plan_parameters_1d_maximum_iterations_without_improvement(self):
        pass

    @property
    def plan_parameters_1d_maximum_water_surface_error_to_abort(self):
        pass

    @property
    def plan_parameters_1d_storage_area_elevation_tolerance(self):
        pass

    @property
    def plan_parameters_1d_theta(self):
        pass

    @property
    def plan_parameters_1d_theta_warmup(self):
        pass

    @property
    def plan_parameters_1d_water_surface_elevation_tolerance(self):
        pass

    @property
    def plan_parameters_1d2d_gate_flow_submergence_decay_exponent(self):
        pass

    @property
    def plan_parameters_1d2d_is_stablity_factor(self):
        pass

    @property
    def plan_parameters_1d2d_ls_stablity_factor(self):
        pass

    @property
    def plan_parameters_1d2d_maximum_number_of_time_slices(self):
        pass

    @property
    def plan_parameters_1d2d_minimum_time_step_for_slicinghours(self):
        pass

    @property
    def plan_parameters_1d2d_number_of_warmup_steps(self):
        pass

    @property
    def plan_parameters_1d2d_warmup_time_step_hours(self):
        pass

    @property
    def plan_parameters_1d2d_weir_flow_submergence_decay_exponent(self):
        pass

    @property
    def plan_parameters_1d2d_maxiter(self):
        pass

    @property
    def plan_parameters_2d_equation_set(self):
        pass

    @property
    def plan_parameters_2d_names(self):
        pass

    @property
    def plan_parameters_2d_volume_tolerance(self):
        pass

    @property
    def plan_parameters_2d_water_surface_tolerance(self):
        pass

    @property
    def meteorology_dss_filename(self):
        pass

    @property
    def meteorology_dss_pathname(self):
        pass

    @property
    def meteorology_data_type(self):
        pass

    @property
    def meteorology_mode(self):
        pass

    @property
    def meteorology_raster_cellsize(self):
        pass

    @property
    def meteorology_source(self):
        pass

    @property
    def meteorology_units(self):
        pass


class GeometryHdfAsset(HdfAsset):
    # class to represent stac asset for geom HDF file associated with model
    def __init__(self, hdf_file: str, **kwargs):
        description = kwargs.get("description", "The HEC-RAS geometry HDF file.")
        super().__init__(hdf_file, RasGeomHdf.open_uri)
        self.hdf_object: RasGeomHdf


class River:

    def __init__(self, river: str, reaches: list[str] = []):
        self.river = river
        self.reaches = reaches

    def gdf(self) -> gpd.GeoDataFrame:
        # TODO: this needs to be defined for the to_gpkg() function to work on GeometryAsset
        pass


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

    def __init__(self, ras_data: list[str], river_reach: str, river: str, reach: str, crs: str, us_xs: XS):
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
                "Deck Dist Width WeirC Skew NumUp NumDn MinLoCord MaxHiCord MaxSubmerge Is_Ogee", 1, self.ras_data
            )
            return data[0].split(",")[position]
        elif self.type == StructureType.INLINE_STRUCTURE:  # 5 = Inline Structure
            data = text_block_from_start_str_length(
                "IW Dist,WD,Coef,Skew,MaxSub,Min_El,Is_Ogee,SpillHt,DesHd", 1, self.ras_data
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
                    f"Unsupported structure type: {int(type)}. Supported structure types are 2, 3, 4, 5, and 6 corresponding to culvert, bridge, multiple openeing, inline structure, lateral structure, respectively"
                )

            structure = Structure(structure_lines, self.river_reach, self.river, self.reach, self.crs, cross_section)
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
