"""Asset instances of HEC-RAS model files."""

import logging
import os
import re
from functools import lru_cache

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import lreshape
from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac import MediaType
from pystac.extensions.projection import ProjectionExtension
from shapely import MultiPolygon, Polygon

from hecstac.common.asset_factory import GenericAsset
from hecstac.common.geometry import reproject_to_wgs84
from hecstac.ras.parser import (
    GeometryFile,
    GeometryHDFFile,
    PlanFile,
    PlanHDFFile,
    ProjectFile,
    QuasiUnsteadyFlowFile,
    SteadyFlowFile,
    UnsteadyFlowFile,
)
from hecstac.ras.utils import is_ras_prj

CURRENT_PLAN = "ras:current_plan"
PLAN_SHORT_ID = "ras:short_plan_id"
TITLE = "ras:title"
UNITS = "ras:units"
VERSION = "ras:version"
PROJECTION = "proj:wkt"

PLAN_FILE = "ras:plan_file"
GEOMETRY_FILE = "ras:geometry_file"
FLOW_FILE = "ras:flow_file"

STEADY_FLOW_FILE = f"ras:steady_{FLOW_FILE}"
QUASI_UNSTEADY_FLOW_FILE = f"ras:quasi_unsteady_{FLOW_FILE}"
UNSTEADY_FLOW_FILE = f"ras:unsteady_{FLOW_FILE}"


PLAN_FILES = f"{PLAN_FILE}s"
GEOMETRY_FILES = f"{GEOMETRY_FILE}s"
STEADY_FLOW_FILES = f"{STEADY_FLOW_FILE}s"
QUASI_UNSTEADY_FLOW_FILES = f"{QUASI_UNSTEADY_FLOW_FILE}s"
UNSTEADY_FLOW_FILES = f"{UNSTEADY_FLOW_FILE}s"

BREACH_LOCATIONS = "ras:breach_locations"
RIVERS = "ras:rivers"
REACHES = "ras:reaches"
JUNCTIONS = "ras:junctions"
CROSS_SECTIONS = "ras:cross_sections"
STRUCTURES = "ras:structures"
STORAGE_AREAS = "ras:storage_areas"
CONNECTIONS = "ras:connections"

HAS_2D = "ras:has_2D_elements"
HAS_1D = "ras:has_1D_elements"

N_PROFILES = "ras:n_profiles"

BOUNDARY_LOCATIONS = "ras:boundary_locations"
REFERENCE_LINES = "ras:reference_lines"

PLAN_INFORMATION_BASE_OUTPUT_INTERVAL = "ras:plan_information_base_output_interval"
PLAN_INFORMATION_COMPUTATION_TIME_STEP_BASE = "ras:plan_information_computation_time_step_base"
PLAN_INFORMATION_FLOW_FILENAME = "ras:plan_information_flow_filename"
PLAN_INFORMATION_GEOMETRY_FILENAME = "ras:plan_information_geometry_filename"
PLAN_INFORMATION_PLAN_FILENAME = "ras:plan_information_plan_filename"
PLAN_INFORMATION_PLAN_NAME = "ras:plan_information_plan_name"
PLAN_INFORMATION_PROJECT_FILENAME = "ras:plan_information_project_filename"
PLAN_INFORMATION_PROJECT_TITLE = "ras:plan_information_project_title"
PLAN_INFORMATION_SIMULATION_END_TIME = "ras:plan_information_simulation_end_time"
PLAN_INFORMATION_SIMULATION_START_TIME = "ras:plan_information_simulation_start_time"
PLAN_PARAMETERS_1D_FLOW_TOLERANCE = "ras:plan_parameters_1d_flow_tolerance"
PLAN_PARAMETERS_1D_MAXIMUM_ITERATIONS = "ras:plan_parameters_1d_maximum_iterations"
PLAN_PARAMETERS_1D_MAXIMUM_ITERATIONS_WITHOUT_IMPROVEMENT = (
    "ras:plan_parameters_1d_maximum_iterations_without_improvement"
)
PLAN_PARAMETERS_1D_MAXIMUM_WATER_SURFACE_ERROR_TO_ABORT = "ras:plan_parameters_1d_maximum_water_surface_error_to_abort"
PLAN_PARAMETERS_1D_STORAGE_AREA_ELEVATION_TOLERANCE = "ras:plan_parameters_1d_storage_area_elevation_tolerance"
PLAN_PARAMETERS_1D_THETA = "ras:plan_parameters_1d_theta"
PLAN_PARAMETERS_1D_THETA_WARMUP = "ras:plan_parameters_1d_theta_warmup"
PLAN_PARAMETERS_1D_WATER_SURFACE_ELEVATION_TOLERANCE = "ras:plan_parameters_1d_water_surface_elevation_tolerance"
PLAN_PARAMETERS_1D2D_GATE_FLOW_SUBMERGENCE_DECAY_EXPONENT = (
    "ras:plan_parameters_1d2d_gate_flow_submergence_decay_exponent"
)
PLAN_PARAMETERS_1D2D_IS_STABLITY_FACTOR = "ras:plan_parameters_1d2d_is_stablity_factor"
PLAN_PARAMETERS_1D2D_LS_STABLITY_FACTOR = "ras:plan_parameters_1d2d_ls_stablity_factor"
PLAN_PARAMETERS_1D2D_MAXIMUM_NUMBER_OF_TIME_SLICES = "ras:plan_parameters_1d2d_maximum_number_of_time_slices"
PLAN_PARAMETERS_1D2D_MINIMUM_TIME_STEP_FOR_SLICINGHOURS = "ras:plan_parameters_1d2d_minimum_time_step_for_slicinghours"
PLAN_PARAMETERS_1D2D_NUMBER_OF_WARMUP_STEPS = "ras:plan_parameters_1d2d_number_of_warmup_steps"
PLAN_PARAMETERS_1D2D_WARMUP_TIME_STEP_HOURS = "ras:plan_parameters_1d2d_warmup_time_step_hours"
PLAN_PARAMETERS_1D2D_WEIR_FLOW_SUBMERGENCE_DECAY_EXPONENT = (
    "ras:plan_parameters_1d2d_weir_flow_submergence_decay_exponent"
)
PLAN_PARAMETERS_1D2D_MAXITER = "ras:plan_parameters_1d2d_maxiter"
PLAN_PARAMETERS_2D_EQUATION_SET = "ras:plan_parameters_2d_equation_set"
PLAN_PARAMETERS_2D_NAMES = "ras:plan_parameters_2d_names"
PLAN_PARAMETERS_2D_VOLUME_TOLERANCE = "ras:plan_parameters_2d_volume_tolerance"
PLAN_PARAMETERS_2D_WATER_SURFACE_TOLERANCE = "ras:plan_parameters_2d_water_surface_tolerance"
METEOROLOGY_DSS_FILENAME = "ras:meteorology_dss_filename"
METEOROLOGY_DSS_PATHNAME = "ras:meteorology_dss_pathname"
METEOROLOGY_DATA_TYPE = "ras:meteorology_data_type"
METEOROLOGY_MODE = "ras:meteorology_mode"
METEOROLOGY_RASTER_CELLSIZE = "ras:meteorology_raster_cellsize"
METEOROLOGY_SOURCE = "ras:meteorology_source"
METEOROLOGY_UNITS = "ras:meteorology_units"


class PrjAsset(GenericAsset):
    """A helper class to delegate .prj files into RAS project or Projection file classes."""

    regex_parse_str = r".+\.prj$"

    def __new__(cls, *args, **kwargs):
        """Delegate to Project or Projection asset."""
        if cls is PrjAsset:  # Ensuring we don't instantiate Parent directly
            href = kwargs.get("href") or args[0]
            is_ras = is_ras_prj(href)
            if is_ras:
                return ProjectAsset(*args, **kwargs)
            else:
                return ProjectionAsset(*args, **kwargs)
        return super().__new__(cls)


class ProjectionAsset(GenericAsset):
    """A geospatial projection file."""

    __roles__ = ["projection-file", MediaType.TEXT]
    __description__ = "A geospatial projection file."
    __file_class__ = None


class ProjectAsset(GenericAsset):
    """HEC-RAS Project file asset."""

    __roles__ = ["project-file", "ras-file"]
    __description__ = "The HEC-RAS project file."
    __file_class__ = ProjectFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[CURRENT_PLAN] = self.file.plan_current
        self._extra_fields[PLAN_FILES] = self.file.plan_files
        self._extra_fields[GEOMETRY_FILES] = self.file.geometry_files
        self._extra_fields[STEADY_FLOW_FILES] = self.file.steady_flow_files
        self._extra_fields[QUASI_UNSTEADY_FLOW_FILES] = self.file.quasi_unsteady_flow_files
        self._extra_fields[UNSTEADY_FLOW_FILES] = self.file.unsteady_flow_files
        return self._extra_fields


class PlanAsset(GenericAsset):
    """HEC-RAS Plan file asset."""

    regex_parse_str = r".+\.p\d{2}$"
    __roles__ = ["plan-file", "ras-file"]
    __description__ = "The plan file which contains a list of associated input files and all simulation options."
    __file_class__ = PlanFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[TITLE] = self.file.plan_title
        self._extra_fields[VERSION] = self.file.plan_version
        self._extra_fields[GEOMETRY_FILE] = self.file.geometry_file
        self._extra_fields[FLOW_FILE] = self.file.flow_file
        self._extra_fields[BREACH_LOCATIONS] = self.file.breach_locations
        return self._extra_fields


class GeometryAsset(GenericAsset):
    """HEC-RAS Geometry file asset."""

    regex_parse_str = r".+\.g\d{2}$"
    __roles__ = ["geometry-file", "ras-file"]
    __description__ = (
        "The geometry file which contains cross-sectional, 2D, hydraulic structures, and other geometric data."
    )
    __file_class__ = GeometryFile
    PROPERTIES_WITH_GDF = ["reaches", "junctions", "cross_sections", "structures"]

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[TITLE] = self.file.geom_title
        self._extra_fields[VERSION] = self.file.geom_version
        self._extra_fields[HAS_1D] = self.file.has_1d
        self._extra_fields[HAS_2D] = self.file.has_2d
        # self._extra_fields[RIVERS] = self.file.rivers
        # self._extra_fields[REACHES] = self.file.reaches
        # self._extra_fields[JUNCTIONS] = self.file.junctions
        # self._extra_fields[CROSS_SECTIONS] = self.file.cross_sections
        # self._extra_fields[STRUCTURES] = self.file.structures
        # self._extra_fields[STORAGE_AREAS] = self.file.storage_areas
        # self._extra_fields[CONNECTIONS] = self.file.connections
        # self._extra_fields[BREACH_LOCATIONS] = self.file.breach_locations
        return self._extra_fields

    @property
    def file(self):
        """Return class to access asset file contents."""
        return self.__file_class__(self.href, self.owner.crs)

    @property
    @lru_cache
    def geometry(self) -> Polygon | MultiPolygon:
        """Retrieves concave hull of cross-sections."""
        return Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])  # TODO:  fill this in.

    @property
    @lru_cache
    def has_1d(self) -> bool:
        """Check if geometry has any river centerlines."""
        return False  # TODO: implement

    @property
    @lru_cache
    def has_2d(self) -> bool:
        """Check if geometry has any 2D areas."""
        return False  # TODO: implement

    @property
    @lru_cache
    def geometry_wgs84(self) -> Polygon | MultiPolygon:
        """Reproject geometry to wgs84."""
        # TODO: this could be generalized to be a function that takes argument for CRS.
        return reproject_to_wgs84(self.geometry, self.crs)


class SteadyFlowAsset(GenericAsset):
    """HEC-RAS Steady Flow file asset."""

    regex_parse_str = r".+\.f\d{2}$"
    __roles__ = ["steady-flow-file", "ras-file"]
    __description__ = "Steady Flow file which contains profile information, flow data, and boundary conditions."
    __file_class__ = SteadyFlowFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[TITLE] = self.file.flow_title
        self._extra_fields[N_PROFILES] = self.file.n_profiles
        return self._extra_fields


class QuasiUnsteadyFlowAsset(GenericAsset):
    """HEC-RAS Quasi-Unsteady Flow file asset."""

    # TODO: implement this class

    regex_parse_str = r".+\.q\d{2}$"
    __roles__ = ["quasi-unsteady-flow-file", "ras-file"]
    __description__ = "Quasi-Unsteady Flow file."
    __file_class__ = QuasiUnsteadyFlowFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[TITLE] = self.file.flow_title
        self._extra_fields[VERSION] = self.file.geom_version
        self._extra_fields[HAS_1D] = self.file.has_1d
        self._extra_fields[HAS_2D] = self.file.has_2d
        self._extra_fields[RIVERS] = self.file.rivers
        self._extra_fields[REACHES] = self.file.reaches
        self._extra_fields[JUNCTIONS] = self.file.junctions
        self._extra_fields[CROSS_SECTIONS] = self.file.cross_sections
        self._extra_fields[STRUCTURES] = self.file.structures
        self._extra_fields[STORAGE_AREAS] = self.file.storage_areas
        self._extra_fields[CONNECTIONS] = self.file.connections
        self._extra_fields[BREACH_LOCATIONS] = self.file.breach_locations
        return self._extra_fields


class UnsteadyFlowAsset(GenericAsset):
    """HEC-RAS Unsteady Flow file asset."""

    regex_parse_str = r".+\.u\d{2}$"
    __roles__ = ["unsteady-flow-file", "ras-file"]
    __description__ = "The unsteady file contains hydrographs, initial conditions, and any flow options."
    __file_class__ = UnsteadyFlowFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[TITLE] = self.file.flow_title
        self._extra_fields[BOUNDARY_LOCATIONS] = self.file.boundary_locations
        self._extra_fields[REFERENCE_LINES] = self.file.reference_lines
        return self._extra_fields


class PlanHdfAsset(GenericAsset):
    """HEC-RAS Plan HDF file asset."""

    regex_parse_str = r".+\.p\d{2}\.hdf$"
    __roles__ = ["ras-file"]
    __description__ = "The HEC-RAS plan HDF file."
    __file_class__ = PlanHDFFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[VERSION] = self.file.flow_title
        self._extra_fields[UNITS] = self.file.units_system
        self._extra_fields[PLAN_INFORMATION_BASE_OUTPUT_INTERVAL] = self.file.plan_information_base_output_interval
        self._extra_fields[PLAN_INFORMATION_COMPUTATION_TIME_STEP_BASE] = (
            self.file.plan_information_computation_time_step_base
        )
        self._extra_fields[PLAN_INFORMATION_FLOW_FILENAME] = self.file.plan_information_flow_filename
        self._extra_fields[PLAN_INFORMATION_GEOMETRY_FILENAME] = self.file.plan_information_geometry_filename
        self._extra_fields[PLAN_INFORMATION_PLAN_FILENAME] = self.file.plan_information_plan_filename
        self._extra_fields[PLAN_INFORMATION_PLAN_NAME] = self.file.plan_information_plan_name
        self._extra_fields[PLAN_INFORMATION_PROJECT_FILENAME] = self.file.plan_information_project_filename
        self._extra_fields[PLAN_INFORMATION_PROJECT_TITLE] = self.file.plan_information_project_title
        self._extra_fields[PLAN_INFORMATION_SIMULATION_END_TIME] = self.file.plan_information_simulation_end_time
        self._extra_fields[PLAN_INFORMATION_SIMULATION_START_TIME] = self.file.plan_information_simulation_start_time
        self._extra_fields[PLAN_PARAMETERS_1D_FLOW_TOLERANCE] = self.file.plan_parameters_1d_flow_tolerance
        self._extra_fields[PLAN_PARAMETERS_1D_MAXIMUM_ITERATIONS] = self.file.plan_parameters_1d_maximum_iterations
        self._extra_fields[PLAN_PARAMETERS_1D_MAXIMUM_ITERATIONS_WITHOUT_IMPROVEMENT] = (
            self.file.plan_parameters_1d_maximum_iterations_without_improvement
        )
        self._extra_fields[PLAN_PARAMETERS_1D_MAXIMUM_WATER_SURFACE_ERROR_TO_ABORT] = (
            self.file.plan_parameters_1d_maximum_water_surface_error_to_abort
        )
        self._extra_fields[PLAN_PARAMETERS_1D_STORAGE_AREA_ELEVATION_TOLERANCE] = (
            self.file.plan_parameters_1d_storage_area_elevation_tolerance
        )
        self._extra_fields[PLAN_PARAMETERS_1D_THETA] = self.file.plan_parameters_1d_theta
        self._extra_fields[PLAN_PARAMETERS_1D_THETA_WARMUP] = self.file.plan_parameters_1d_theta_warmup
        self._extra_fields[PLAN_PARAMETERS_1D_WATER_SURFACE_ELEVATION_TOLERANCE] = (
            self.file.plan_parameters_1d_water_surface_elevation_tolerance
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_GATE_FLOW_SUBMERGENCE_DECAY_EXPONENT] = (
            self.file.plan_parameters_1d2d_gate_flow_submergence_decay_exponent
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_IS_STABLITY_FACTOR] = self.file.plan_parameters_1d2d_is_stablity_factor
        self._extra_fields[PLAN_PARAMETERS_1D2D_LS_STABLITY_FACTOR] = self.file.plan_parameters_1d2d_ls_stablity_factor
        self._extra_fields[PLAN_PARAMETERS_1D2D_MAXIMUM_NUMBER_OF_TIME_SLICES] = (
            self.file.plan_parameters_1d2d_maximum_number_of_time_slices
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_MINIMUM_TIME_STEP_FOR_SLICINGHOURS] = (
            self.file.plan_parameters_1d2d_minimum_time_step_for_slicinghours
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_NUMBER_OF_WARMUP_STEPS] = (
            self.file.plan_parameters_1d2d_number_of_warmup_steps
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_WARMUP_TIME_STEP_HOURS] = (
            self.file.plan_parameters_1d2d_warmup_time_step_hours
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_WEIR_FLOW_SUBMERGENCE_DECAY_EXPONENT] = (
            self.file.plan_parameters_1d2d_weir_flow_submergence_decay_exponent
        )
        self._extra_fields[PLAN_PARAMETERS_1D2D_MAXITER] = self.file.plan_parameters_1d2d_maxiter
        self._extra_fields[PLAN_PARAMETERS_2D_EQUATION_SET] = self.file.plan_parameters_2d_equation_set
        self._extra_fields[PLAN_PARAMETERS_2D_NAMES] = self.file.plan_parameters_2d_names
        self._extra_fields[PLAN_PARAMETERS_2D_VOLUME_TOLERANCE] = self.file.plan_parameters_2d_volume_tolerance
        self._extra_fields[PLAN_PARAMETERS_2D_WATER_SURFACE_TOLERANCE] = (
            self.file.plan_parameters_2d_water_surface_tolerance
        )
        self._extra_fields[METEOROLOGY_DSS_FILENAME] = self.file.meteorology_dss_filename
        self._extra_fields[METEOROLOGY_DSS_PATHNAME] = self.file.meteorology_dss_pathname
        self._extra_fields[METEOROLOGY_DATA_TYPE] = self.file.meteorology_data_type
        self._extra_fields[METEOROLOGY_MODE] = self.file.meteorology_mode
        self._extra_fields[METEOROLOGY_RASTER_CELLSIZE] = self.file.meteorology_raster_cellsize
        self._extra_fields[METEOROLOGY_SOURCE] = self.file.meteorology_source
        self._extra_fields[METEOROLOGY_UNITS] = self.file.meteorology_units
        return self._extra_fields


class GeometryHdfAsset(GenericAsset):
    """HEC-RAS Geometry HDF file asset."""

    regex_parse_str = r".+\.g\d{2}\.hdf$"
    __roles__ = ["geometry-hdf-file"]
    __description__ = "The HEC-RAS geometry HDF file."
    __file_class__ = GeometryHDFFile

    @GenericAsset.extra_fields.getter
    def extra_fields(self) -> dict:
        """Return extra fields with added dynamic keys/values."""
        self._extra_fields[VERSION] = self.file.file_version
        self._extra_fields[UNITS] = self.file.units_system
        self._extra_fields[PROJECTION] = self.owner.crs.to_wkt()
        self._extra_fields[UNITS] = self.file.units_system
        return self._extra_fields

    @property
    def reference_lines(self) -> list[gpd.GeoDataFrame] | None:
        """Docstring."""  # TODO: fill out
        if self.hdf_object.reference_lines is not None and not self.hdf_object.reference_lines.empty:
            return list(self.file.reference_lines["refln_name"])

    @property
    @lru_cache
    def has_2d(self) -> bool:
        """Check if the geometry asset has 2d geometry."""
        try:
            logging.debug(f"reading mesh areas using crs {self.crs}...")

            if self.hdf_object.mesh_areas(self.crs):
                return True
        except ValueError:
            logging.warning(f"No mesh areas found for {self.href}")
            return False

    @property
    @lru_cache
    def has_1d(self) -> bool:
        """Check if the geometry asset has 2d geometry."""
        return False  # TODO: implement

    @property
    @lru_cache
    def geometry(self, crs: CRS) -> Polygon | MultiPolygon:
        """Retrieves concave hull of cross-sections."""
        return self.hdf_object.mesh_areas(crs)

    @property
    @lru_cache
    def geometry_wgs84(self) -> Polygon | MultiPolygon:
        """Reproject geometry to wgs84."""
        # TODO: this could be generalized to be a function that takes argument for CRS.
        return reproject_to_wgs84(self.geometry, self.crs)

    def _plot_mesh_areas(self, ax, mesh_polygons: gpd.GeoDataFrame) -> list[Line2D]:
        """Plot mesh areas on the given axes."""
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
        """Plot breaklines on the given axes."""
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
        """Plot boundary condition lines on the given axes."""
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
        if filepath.startswith("s3://"):
            media_type = "image/png"
        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Thumbnail file not found: {filepath}")
            media_type = "image/png"

        return GenericAsset(
            href=filepath,
            title=filepath.split("/")[-1],
            description="Thumbnail image for the model",
            media_type=media_type,
            roles=["thumbnail"],
            extra_fields=None,
        )

    def thumbnail(
        self,
        layers: list,
        title: str = "Model_Thumbnail",
        thumbnail_dest: str = None,
    ):
        """
        Create a thumbnail figure for a geometry hdf file, includingvarious geospatial layers such as USGS gages, mesh areas, breaklines, and boundary condition (BC) lines.

        Parameters
        ----------
        layers : list
            A list of model layers to include in the thumbnail plot.
            Options include "usgs_gages", "mesh_areas", "breaklines", and "bc_lines".
        title : str, optional
            Title of the figure, by default "Model Thumbnail".
        thumbnail_dest : str, optional
            Directory for created thumbnails. If None then thumbnails will be exported to same level as the item.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        legend_handles = []

        for layer in layers:
            try:
                if layer == "mesh_areas":
                    mesh_areas_data = self.hdf_object.mesh_cells
                    mesh_areas_geo = mesh_areas_data.set_crs(self.crs)
                    legend_handles += self._plot_mesh_areas(ax, mesh_areas_geo)
                elif layer == "breaklines":
                    breaklines_data = self.hdf_object.breaklines
                    breaklines_data_geo = breaklines_data.set_crs(self.crs)
                    legend_handles += self._plot_breaklines(ax, breaklines_data_geo)
                elif layer == "bc_lines":
                    bc_lines_data = self.hdf_object.bc_lines
                    bc_lines_data_geo = bc_lines_data.set_crs(self.crs)
                    legend_handles += self._plot_bc_lines(ax, bc_lines_data_geo)
            except Exception as e:
                logging.warning(f"Warning: Failed to process layer '{layer}' for {self.href}: {e}")

        # Add OpenStreetMap basemap
        ctx.add_basemap(
            ax,
            crs=self.crs,
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=0.4,
        )
        ax.set_title(f"{title} - {os.path.basename(self.href)}", fontsize=15)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1, 0.5))

        hdf_ext = os.path.basename(self.href).split(".")[-2]
        filename = f"thumbnail_{hdf_ext}.png"
        base_dir = os.path.dirname(thumbnail_dest)
        filepath = os.path.join(base_dir, filename)

        if filepath.startswith("s3://"):
            pass
            # TODO add thumbnail s3 functionality
            # img_data = io.BytesIO()
            # fig.savefig(img_data, format="png", bbox_inches="tight")
            # img_data.seek(0)
            # save_bytes_s3(img_data, filepath)
        else:
            os.makedirs(base_dir, exist_ok=True)
            fig.savefig(filepath, dpi=80, bbox_inches="tight")
        return self._add_thumbnail_asset(filepath)


class RunFileAsset(GenericAsset):
    """Run file asset for steady flow analysis."""

    regex_parse_str = r".+\.r\d{2}$"
    __roles__ = ["run-file", "ras-file", MediaType.TEXT]
    __description__ = "Run file for steady flow analysis which contains all the necessary input data required for the RAS computational engine."
    __file_class__ = None


class ComputationalLevelOutputAsset(GenericAsset):
    """Computational Level Output asset."""

    regex_parse_str = r".+\.hyd\d{2}$"
    __roles__ = ["computational-level-output-file", "ras-file", MediaType.TEXT]
    __description__ = "Detailed Computational Level output file."
    __file_class__ = None


class GeometricPreprocessorAsset(GenericAsset):
    """Geometric Pre-Processor asset."""

    regex_parse_str = r".+\.c\d{2}$"
    __roles__ = ["geometric-preprocessor", "ras-file", MediaType.TEXT]
    __description__ = "Geometric Pre-Processor output file containing hydraulic properties, rating curves, and more."
    __file_class__ = None  # TODO:  make a generic parent for these.


class BoundaryConditionAsset(GenericAsset):
    """Boundary Condition asset."""

    regex_parse_str = r".+\.b\d{2}$"
    __roles__ = ["boundary-condition-file", "ras-file", MediaType.TEXT]
    __description__ = "Boundary Condition file."
    __file_class__ = None


class UnsteadyFlowLogAsset(GenericAsset):
    """Unsteady Flow Log asset."""

    regex_parse_str = r".+\.bco\d{2}$"
    __roles__ = ["unsteady-flow-log-file", "ras-file", MediaType.TEXT]
    __description__ = "Unsteady Flow Log output file."
    __file_class__ = None


class SedimentDataAsset(GenericAsset):
    """Sediment Data asset."""

    regex_parse_str = r".+\.s\d{2}$"
    __roles__ = ["sediment-data-file", "ras-file", MediaType.TEXT]
    __description__ = "Sediment data file containing flow data, boundary conditions, and sediment data."
    __file_class__ = None


class HydraulicDesignAsset(GenericAsset):
    """Hydraulic Design asset."""

    regex_parse_str = r".+\.h\d{2}$"
    __roles__ = ["hydraulic-design-file", "ras-file", MediaType.TEXT]
    __description__ = "Hydraulic Design data file."
    __file_class__ = None


class WaterQualityAsset(GenericAsset):
    """Water Quality asset."""

    regex_parse_str = r".+\.w\d{2}$"
    __roles__ = ["water-quality-file", "ras-file", MediaType.TEXT]
    __description__ = "Water Quality file containing temperature boundary conditions and meteorological data."
    __file_class__ = None


class SedimentTransportCapacityAsset(GenericAsset):
    """Sediment Transport Capacity asset."""

    regex_parse_str = r".+\.SedCap\d{2}$"
    __roles__ = ["sediment-transport-capacity-file", "ras-file", MediaType.TEXT]
    __description__ = "Sediment Transport Capacity data."
    __file_class__ = None


class XSOutputAsset(GenericAsset):
    """Cross Section Output asset."""

    regex_parse_str = r".+\.SedXS\d{2}$"
    __roles__ = ["xs-output-file", "ras-file", MediaType.TEXT]
    __description__ = "Cross section output file."
    __file_class__ = None


class XSOutputHeaderAsset(GenericAsset):
    """Cross Section Output Header asset."""

    regex_parse_str = r".+\.SedHeadXS\d{2}$"
    __roles__ = ["xs-output-header-file", "ras-file", MediaType.TEXT]
    __description__ = "Header file for the cross section output."
    __file_class__ = None


class WaterQualityRestartAsset(GenericAsset):
    """Water Quality Restart asset."""

    regex_parse_str = r".+\.wqrst\d{2}$"
    __roles__ = ["water-quality-restart-file", "ras-file", MediaType.TEXT]
    __description__ = "The water quality restart file."
    __file_class__ = None


class SedimentOutputAsset(GenericAsset):
    """Sediment Output asset."""

    regex_parse_str = r".+\.sed$"
    __roles__ = ["sediment-output-file", "ras-file", MediaType.TEXT]
    __description__ = "Detailed sediment output file."
    __file_class__ = None


class BinaryLogAsset(GenericAsset):
    """Binary Log asset."""

    regex_parse_str = r".+\.blf$"
    __roles__ = ["binary-log-file", "ras-file", MediaType.TEXT]
    __description__ = "Binary Log file."
    __file_class__ = None


class DSSAsset(GenericAsset):
    """DSS asset."""

    regex_parse_str = r".+\.dss$"
    __roles__ = ["ras-dss", "ras-file", MediaType.TEXT]
    __description__ = "The DSS file contains results and other simulation information."
    __file_class__ = None


class LogAsset(GenericAsset):
    """Log asset."""

    regex_parse_str = r".+\.log$"
    __roles__ = ["ras-log", "ras-file", MediaType.TEXT]
    __description__ = "The log file contains information related to simulation processes."
    __file_class__ = None


class RestartAsset(GenericAsset):
    """Restart file asset."""

    regex_parse_str = r".+\.rst$"
    __roles__ = ["restart-file", "ras-file", MediaType.TEXT]
    __description__ = "Restart file for resuming simulation runs."
    __file_class__ = None


class SiamInputAsset(GenericAsset):
    """SIAM Input Data file asset."""

    regex_parse_str = r".+\.SiamInput$"
    __roles__ = ["siam-input-file", "ras-file", MediaType.TEXT]
    __description__ = "SIAM Input Data file."
    __file_class__ = None


class SiamOutputAsset(GenericAsset):
    """SIAM Output Data file asset."""

    regex_parse_str = r".+\.SiamOutput$"
    __roles__ = ["siam-output-file", "ras-file", MediaType.TEXT]
    __description__ = "SIAM Output Data file."
    __file_class__ = None


class WaterQualityLogAsset(GenericAsset):
    """Water Quality Log file asset."""

    regex_parse_str = r".+\.bco$"
    __roles__ = ["water-quality-log", "ras-file", MediaType.TEXT]
    __description__ = "Water quality log file."
    __file_class__ = None


class ColorScalesAsset(GenericAsset):
    """Color Scales file asset."""

    regex_parse_str = r".+\.color-scales$"
    __roles__ = ["color-scales", "ras-file", MediaType.TEXT]
    __description__ = "File that contains the water quality color scale."
    __file_class__ = None


class ComputationalMessageAsset(GenericAsset):
    """Computational Message file asset."""

    regex_parse_str = r".+\.comp-msgs.txt$"
    __roles__ = ["computational-message-file", "ras-file", MediaType.TEXT]
    __description__ = "Computational Message text file which contains messages from the computation process."
    __file_class__ = None


class UnsteadyRunFileAsset(GenericAsset):
    """Run file for Unsteady Flow asset."""

    regex_parse_str = r".+\.x\d{2}$"
    __roles__ = ["run-file", "ras-file", MediaType.TEXT]
    __description__ = "Run file for Unsteady Flow simulations."
    __file_class__ = None


class OutputFileAsset(GenericAsset):
    """Output RAS file asset."""

    regex_parse_str = r".+\.o\d{2}$"
    __roles__ = ["output-file", "ras-file", MediaType.TEXT]
    __description__ = "Output RAS file which contains all computed results."
    __file_class__ = None


class InitialConditionsFileAsset(GenericAsset):
    """Initial Conditions file asset."""

    regex_parse_str = r".+\.IC\.O\d{2}$"
    __roles__ = ["initial-conditions-file", "ras-file", MediaType.TEXT]
    __description__ = "Initial conditions file for unsteady flow plan."
    __file_class__ = None


class PlanRestartFileAsset(GenericAsset):
    """Restart file for Unsteady Flow Plan asset."""

    regex_parse_str = r".+\.p\d{2}\.rst$"
    __roles__ = ["restart-file", "ras-file", MediaType.TEXT]
    __description__ = "Restart file for unsteady flow plan."
    __file_class__ = None


class RasMapperFileAsset(GenericAsset):
    """RAS Mapper file asset."""

    regex_parse_str = r".+\.rasmap$"
    __roles__ = ["ras-mapper-file", "ras-file", MediaType.TEXT]
    __description__ = "RAS Mapper file."
    __file_class__ = None


class RasMapperBackupFileAsset(GenericAsset):
    """Backup RAS Mapper file asset."""

    regex_parse_str = r".+\.rasmap\.backup$"
    __roles__ = ["ras-mapper-file", "ras-file", MediaType.TEXT]
    __description__ = "Backup RAS Mapper file."
    __file_class__ = None


class RasMapperOriginalFileAsset(GenericAsset):
    """Original RAS Mapper file asset."""

    regex_parse_str = r".+\.rasmap\.original$"
    __roles__ = ["ras-mapper-file", "ras-file", MediaType.TEXT]
    __description__ = "Original RAS Mapper file."
    __file_class__ = None


class MiscTextFileAsset(GenericAsset):
    """Miscellaneous Text file asset."""

    regex_parse_str = r".+\.txt$"
    __roles__ = [MediaType.TEXT]
    __description__ = "Miscellaneous text file."
    __file_class__ = None


class MiscXMLFileAsset(GenericAsset):
    """Miscellaneous XML file asset."""

    regex_parse_str = r".+\.xml$"
    __roles__ = [MediaType.XML]
    __description__ = "Miscellaneous XML file."
    __file_class__ = None


RAS_ASSET_CLASSES = [
    PrjAsset,
    PlanAsset,
    GeometryAsset,
    SteadyFlowAsset,
    # QuasiUnsteadyFlowAsset,
    UnsteadyFlowAsset,
    PlanHdfAsset,
    GeometryHdfAsset,
    RunFileAsset,
    ComputationalLevelOutputAsset,
    GeometricPreprocessorAsset,
    BoundaryConditionAsset,
    UnsteadyFlowLogAsset,
    SedimentDataAsset,
    HydraulicDesignAsset,
    WaterQualityAsset,
    SedimentTransportCapacityAsset,
    XSOutputAsset,
    XSOutputHeaderAsset,
    WaterQualityRestartAsset,
    SedimentOutputAsset,
    BinaryLogAsset,
    DSSAsset,
    LogAsset,
    RestartAsset,
    SiamInputAsset,
    SiamOutputAsset,
    WaterQualityLogAsset,
    ColorScalesAsset,
    ComputationalMessageAsset,
    UnsteadyRunFileAsset,
    OutputFileAsset,
    InitialConditionsFileAsset,
    PlanRestartFileAsset,
    RasMapperFileAsset,
    RasMapperBackupFileAsset,
    RasMapperOriginalFileAsset,
    MiscTextFileAsset,
    MiscXMLFileAsset,
]

RAS_EXTENSION_MAPPING = {re.compile(cls.regex_parse_str, re.IGNORECASE): cls for cls in RAS_ASSET_CLASSES}
