import logging
import os
import re

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pystac import MediaType
from pyproj import CRS
from pyproj.exceptions import CRSError
from hecstac.common.asset_factory import GenericAsset
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


class ProjectAsset(GenericAsset):
    """HEC-RAS Project file asset."""

    regex_parse_str = r".+\.prj$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["project-file", "ras-file"]
        description = kwargs.get("description", "The HEC-RAS project file.")

        super().__init__(href, roles=roles, description=description, *args, **kwargs)

        self.href = href
        self.pf = ProjectFile(self.href)
        self.extra_fields = {
            key: value
            for key, value in {
                CURRENT_PLAN: self.pf.plan_current,
                PLAN_FILES: self.pf.plan_files,
                GEOMETRY_FILES: self.pf.geometry_files,
                STEADY_FLOW_FILES: self.pf.steady_flow_files,
                QUASI_UNSTEADY_FLOW_FILES: self.pf.quasi_unsteady_flow_files,
                UNSTEADY_FLOW_FILES: self.pf.unsteady_flow_files,
            }.items()
            if value
        }


class PlanAsset(GenericAsset):
    """HEC-RAS Plan file asset."""

    regex_parse_str = r".+\.p\d{2}$"

    def __init__(self, href: str, **kwargs):
        roles = kwargs.get("roles", []) + ["plan-file", "ras-file"]
        description = kwargs.get(
            "description",
            "The plan file which contains a list of associated input files and all simulation options.",
        )

        super().__init__(href, roles=roles, description=description, **kwargs)

        self.href = href
        self.planf = PlanFile(self.href)
        self.extra_fields = {
            key: value
            for key, value in {
                TITLE: self.planf.plan_title,
                VERSION: self.planf.plan_version,
                GEOMETRY_FILE: self.planf.geometry_file,
                FLOW_FILE: self.planf.flow_file,
                BREACH_LOCATIONS: self.planf.breach_locations,
            }.items()
            if value
        }


class GeometryAsset(GenericAsset):
    """HEC-RAS Geometry file asset."""

    regex_parse_str = r".+\.g\d{2}$"
    PROPERTIES_WITH_GDF = ["reaches", "junctions", "cross_sections", "structures"]

    def __init__(self, href: str, crs: str = None, **kwargs):
        # self.pyproj_crs = self.validate_crs(crs)
        self.crs = crs
        roles = kwargs.get("roles", []) + ["geometry-file", "ras-file"]
        description = kwargs.get(
            "description",
            "The geometry file which contains cross-sectional, 2D, hydraulic structures, and other geometric data",
        )

        super().__init__(href, roles=roles, description=description, **kwargs)

        self.href = href
        self.geomf = GeometryFile(self.href, self.crs)
        self.extra_fields = {
            key: value
            for key, value in {
                TITLE: self.geomf.geom_title,
                VERSION: self.geomf.geom_version,
                HAS_1D: self.geomf.has_1d,
                HAS_2D: self.geomf.has_2d,
                # RIVERS: self.geomf.rivers,
                # REACHES: self.geomf.reaches,
                # JUNCTIONS: self.geomf.junctions,
                # CROSS_SECTIONS: self.geomf.cross_sections,
                # STRUCTURES: self.geomf.structures,
                # STORAGE_AREAS: self.geomf.storage_areas, #TODO: fix this
                # CONNECTIONS: self.geomf.connections,#TODO: fix this
                # BREACH_LOCATIONS: self.planf.breach_locations,
            }.items()
            if value
        }


class SteadyFlowAsset(GenericAsset):
    """HEC-RAS Steady Flow file asset."""

    regex_parse_str = r".+\.f\d{2}$"

    def __init__(self, href: str, **kwargs):
        roles = kwargs.get("roles", []) + ["steady-flow-file", "ras-file"]
        description = kwargs.get(
            "description",
            "Steady Flow file which contains profile information, flow data, and boundary conditions.",
        )

        super().__init__(href, roles=roles, description=description, **kwargs)

        self.href = href
        self.flowf = SteadyFlowFile(self.href)
        self.extra_fields = {
            key: value
            for key, value in {
                TITLE: self.flowf.flow_title,
                N_PROFILES: self.flowf.n_profiles,
            }.items()
            if value
        }


class QuasiUnsteadyFlowAsset(GenericAsset):
    """HEC-RAS Quasi-Unsteady Flow file asset."""

    # TODO: implement this class

    regex_parse_str = r".+\.q\d{2}$"

    def __init__(self, href: str, **kwargs):
        roles = kwargs.get("roles", []) + ["quasi-unsteady-flow-file", "ras-file"]
        description = kwargs.get("description", "Quasi-Unsteady Flow file.")

        super().__init__(href, roles=roles, description=description, **kwargs)

        self.href = href
        self.flowf = QuasiUnsteadyFlowFile(self.href)
        self.extra_fields = {
            key: value
            for key, value in {
                TITLE: self.flowf.flow_title,
            }.items()
            if value
        }


class UnsteadyFlowAsset(GenericAsset):
    """HEC-RAS Unsteady Flow file asset."""

    regex_parse_str = r".+\.u\d{2}$"

    def __init__(self, href: str, **kwargs):
        roles = kwargs.get("roles", []) + ["unsteady-flow-file", "ras-file"]
        description = kwargs.get(
            "description",
            "The unsteady file contains hydrographs, initial conditions, and any flow options.",
        )

        super().__init__(href, roles=roles, description=description, **kwargs)

        self.href = href
        self.flowf = UnsteadyFlowFile(self.href)
        self.extra_fields = {
            key: value
            for key, value in {
                TITLE: self.flowf.flow_title,
                BOUNDARY_LOCATIONS: self.flowf.boundary_locations,
                REFERENCE_LINES: self.flowf.reference_lines,
            }.items()
            if value
        }


class PlanHdfAsset(GenericAsset):
    """HEC-RAS Plan HDF file asset."""

    regex_parse_str = r".+\.p\d{2}\.hdf$"

    def __init__(self, href: str, **kwargs):
        roles = kwargs.get("roles", []) + ["ras-file"]
        description = kwargs.get("description", "The HEC-RAS plan HDF file.")

        super().__init__(href, roles=roles, description=description, **kwargs)

        self.hdf_object = PlanHDFFile(self.href)
        self.extra_fields = {
            key: value
            for key, value in {
                VERSION: self.hdf_object.file_version,
                UNITS: self.hdf_object.units_system,
                PLAN_INFORMATION_BASE_OUTPUT_INTERVAL: self.hdf_object.plan_information_base_output_interval,
                PLAN_INFORMATION_COMPUTATION_TIME_STEP_BASE: self.hdf_object.plan_information_computation_time_step_base,
                PLAN_INFORMATION_FLOW_FILENAME: self.hdf_object.plan_information_flow_filename,
                PLAN_INFORMATION_GEOMETRY_FILENAME: self.hdf_object.plan_information_geometry_filename,
                PLAN_INFORMATION_PLAN_FILENAME: self.hdf_object.plan_information_plan_filename,
                PLAN_INFORMATION_PLAN_NAME: self.hdf_object.plan_information_plan_name,
                PLAN_INFORMATION_PROJECT_FILENAME: self.hdf_object.plan_information_project_filename,
                PLAN_INFORMATION_PROJECT_TITLE: self.hdf_object.plan_information_project_title,
                PLAN_INFORMATION_SIMULATION_END_TIME: self.hdf_object.plan_information_simulation_end_time,
                PLAN_INFORMATION_SIMULATION_START_TIME: self.hdf_object.plan_information_simulation_start_time,
                PLAN_PARAMETERS_1D_FLOW_TOLERANCE: self.hdf_object.plan_parameters_1d_flow_tolerance,
                PLAN_PARAMETERS_1D_MAXIMUM_ITERATIONS: self.hdf_object.plan_parameters_1d_maximum_iterations,
                PLAN_PARAMETERS_1D_MAXIMUM_ITERATIONS_WITHOUT_IMPROVEMENT: self.hdf_object.plan_parameters_1d_maximum_iterations_without_improvement,
                PLAN_PARAMETERS_1D_MAXIMUM_WATER_SURFACE_ERROR_TO_ABORT: self.hdf_object.plan_parameters_1d_maximum_water_surface_error_to_abort,
                PLAN_PARAMETERS_1D_STORAGE_AREA_ELEVATION_TOLERANCE: self.hdf_object.plan_parameters_1d_storage_area_elevation_tolerance,
                PLAN_PARAMETERS_1D_THETA: self.hdf_object.plan_parameters_1d_theta,
                PLAN_PARAMETERS_1D_THETA_WARMUP: self.hdf_object.plan_parameters_1d_theta_warmup,
                PLAN_PARAMETERS_1D_WATER_SURFACE_ELEVATION_TOLERANCE: self.hdf_object.plan_parameters_1d_water_surface_elevation_tolerance,
                PLAN_PARAMETERS_1D2D_GATE_FLOW_SUBMERGENCE_DECAY_EXPONENT: self.hdf_object.plan_parameters_1d2d_gate_flow_submergence_decay_exponent,
                PLAN_PARAMETERS_1D2D_IS_STABLITY_FACTOR: self.hdf_object.plan_parameters_1d2d_is_stablity_factor,
                PLAN_PARAMETERS_1D2D_LS_STABLITY_FACTOR: self.hdf_object.plan_parameters_1d2d_ls_stablity_factor,
                PLAN_PARAMETERS_1D2D_MAXIMUM_NUMBER_OF_TIME_SLICES: self.hdf_object.plan_parameters_1d2d_maximum_number_of_time_slices,
                PLAN_PARAMETERS_1D2D_MINIMUM_TIME_STEP_FOR_SLICINGHOURS: self.hdf_object.plan_parameters_1d2d_minimum_time_step_for_slicinghours,
                PLAN_PARAMETERS_1D2D_NUMBER_OF_WARMUP_STEPS: self.hdf_object.plan_parameters_1d2d_number_of_warmup_steps,
                PLAN_PARAMETERS_1D2D_WARMUP_TIME_STEP_HOURS: self.hdf_object.plan_parameters_1d2d_warmup_time_step_hours,
                PLAN_PARAMETERS_1D2D_WEIR_FLOW_SUBMERGENCE_DECAY_EXPONENT: self.hdf_object.plan_parameters_1d2d_weir_flow_submergence_decay_exponent,
                PLAN_PARAMETERS_1D2D_MAXITER: self.hdf_object.plan_parameters_1d2d_maxiter,
                PLAN_PARAMETERS_2D_EQUATION_SET: self.hdf_object.plan_parameters_2d_equation_set,
                PLAN_PARAMETERS_2D_NAMES: self.hdf_object.plan_parameters_2d_names,
                PLAN_PARAMETERS_2D_VOLUME_TOLERANCE: self.hdf_object.plan_parameters_2d_volume_tolerance,
                PLAN_PARAMETERS_2D_WATER_SURFACE_TOLERANCE: self.hdf_object.plan_parameters_2d_water_surface_tolerance,
                METEOROLOGY_DSS_FILENAME: self.hdf_object.meteorology_dss_filename,
                METEOROLOGY_DSS_PATHNAME: self.hdf_object.meteorology_dss_pathname,
                METEOROLOGY_DATA_TYPE: self.hdf_object.meteorology_data_type,
                METEOROLOGY_MODE: self.hdf_object.meteorology_mode,
                METEOROLOGY_RASTER_CELLSIZE: self.hdf_object.meteorology_raster_cellsize,
                METEOROLOGY_SOURCE: self.hdf_object.meteorology_source,
                METEOROLOGY_UNITS: self.hdf_object.meteorology_units,
            }.items()
            if value
        }


class GeometryHdfAsset(GenericAsset):
    """HEC-RAS Geometry HDF file asset."""

    regex_parse_str = r".+\.g\d{2}\.hdf$"

    def __init__(self, href: str, crs: str = None, **kwargs):
        roles = kwargs.get("roles", []) + ["geometry-hdf-file"]
        description = kwargs.get("description", "The HEC-RAS geometry HDF file.")

        super().__init__(href, roles=roles, description=description, **kwargs)
        self.hdf_object = GeometryHDFFile(self.href)
        self.crs = crs
        self.has_2d = None
        if self.crs is None:
            try:
                self.crs = CRS.from_user_input(self.hdf_object.projection)
                logging.info(f"crs has been set to {self.crs}")
            except CRSError:
                logging.warning(f"Could not extract crs from {self.href}")

        self.extra_fields = {
            key: value
            for key, value in {
                VERSION: self.hdf_object.file_version,
                UNITS: self.hdf_object.units_system,
                PROJECTION: self.crs.to_wkt() if self.crs is not None else None,
                REFERENCE_LINES: (
                    list(self.hdf_object.reference_lines["refln_name"])
                    if self.hdf_object.reference_lines is not None and not self.hdf_object.reference_lines.empty
                    else None
                ),
            }.items()
            if value
        }

    @property
    def check_2d(self):
        """Check if the geometry asset has 2d geometry, if yes then return True and set has_2d to True."""
        try:
            logging.debug(f"reading mesh areas using crs {self.crs}...")

            if self.hdf_object.mesh_areas(self.crs):
                self.has_2d = True
                return True
        except ValueError:
            logging.warning(f"No mesh areas found for {self.href}")
            self.has_2d = False
            return False

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
            title=filename.split(".")[0],
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
        """Create a thumbnail figure for a geometry hdf file, including
        various geospatial layers such as USGS gages, mesh areas,
        breaklines, and boundary condition (BC) lines.

        Parameters
        ----------
        layers : list
            A list of model layers to include in the thumbnail plot.
            Options include "usgs_gages", "mesh_areas", "breaklines", and "bc_lines".
        title : str, optional
            Title of the figure, by default "Model Thumbnail".
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

    def __init__(self, href: str, *args, **kwargs):
        roles = ["run-file", "ras-file", MediaType.TEXT]
        description = "Run file for steady flow analysis which contains all the necessary input data required for the RAS computational engine."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class ComputationalLevelOutputAsset(GenericAsset):
    """Computational Level Output asset."""

    regex_parse_str = r".+\.hyd\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["computational-level-output-file", "ras-file", MediaType.TEXT]
        description = "Detailed Computational Level output file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class GeometricPreprocessorAsset(GenericAsset):
    """Geometric Pre-Processor asset."""

    regex_parse_str = r".+\.c\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["geometric-preprocessor", "ras-file", MediaType.TEXT]
        description = "Geometric Pre-Processor output file containing hydraulic properties, rating curves, and more."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class BoundaryConditionAsset(GenericAsset):
    """Boundary Condition asset."""

    regex_parse_str = r".+\.b\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["boundary-condition-file", "ras-file", MediaType.TEXT]
        description = "Boundary Condition file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class UnsteadyFlowLogAsset(GenericAsset):
    """Unsteady Flow Log asset."""

    regex_parse_str = r".+\.bco\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["unsteady-flow-log-file", "ras-file", MediaType.TEXT]
        description = "Unsteady Flow Log output file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class SedimentDataAsset(GenericAsset):
    """Sediment Data asset."""

    regex_parse_str = r".+\.s\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["sediment-data-file", "ras-file", MediaType.TEXT]
        description = "Sediment data file containing flow data, boundary conditions, and sediment data."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class HydraulicDesignAsset(GenericAsset):
    """Hydraulic Design asset."""

    regex_parse_str = r".+\.h\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["hydraulic-design-file", "ras-file", MediaType.TEXT]
        description = "Hydraulic Design data file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class WaterQualityAsset(GenericAsset):
    """Water Quality asset."""

    regex_parse_str = r".+\.w\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["water-quality-file", "ras-file", MediaType.TEXT]
        description = "Water Quality file containing temperature boundary conditions and meteorological data."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class SedimentTransportCapacityAsset(GenericAsset):
    """Sediment Transport Capacity asset."""

    regex_parse_str = r".+\.SedCap\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["sediment-transport-capacity-file", "ras-file", MediaType.TEXT]
        description = "Sediment Transport Capacity data."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class XSOutputAsset(GenericAsset):
    """Cross Section Output asset."""

    regex_parse_str = r".+\.SedXS\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["xs-output-file", "ras-file", MediaType.TEXT]
        description = "Cross section output file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class XSOutputHeaderAsset(GenericAsset):
    """Cross Section Output Header asset."""

    regex_parse_str = r".+\.SedHeadXS\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["xs-output-header-file", "ras-file", MediaType.TEXT]
        description = "Header file for the cross section output."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class WaterQualityRestartAsset(GenericAsset):
    """Water Quality Restart asset."""

    regex_parse_str = r".+\.wqrst\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["water-quality-restart-file", "ras-file", MediaType.TEXT]
        description = "The water quality restart file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class SedimentOutputAsset(GenericAsset):
    """Sediment Output asset."""

    regex_parse_str = r".+\.sed$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["sediment-output-file", "ras-file", MediaType.TEXT]
        description = "Detailed sediment output file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class BinaryLogAsset(GenericAsset):
    """Binary Log asset."""

    regex_parse_str = r".+\.blf$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["binary-log-file", "ras-file", MediaType.TEXT]
        description = "Binary Log file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class DSSAsset(GenericAsset):
    """DSS asset."""

    regex_parse_str = r".+\.dss$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["ras-dss", "ras-file", MediaType.TEXT]
        description = "The DSS file contains results and other simulation information."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class LogAsset(GenericAsset):
    """Log asset."""

    regex_parse_str = r".+\.log$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["ras-log", "ras-file", MediaType.TEXT]
        description = "The log file contains information related to simulation processes."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class RestartAsset(GenericAsset):
    """Restart file asset."""

    regex_parse_str = r".+\.rst$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["restart-file", "ras-file", MediaType.TEXT]
        description = "Restart file for resuming simulation runs."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class SiamInputAsset(GenericAsset):
    """SIAM Input Data file asset."""

    regex_parse_str = r".+\.SiamInput$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["siam-input-file", "ras-file", MediaType.TEXT]
        description = "SIAM Input Data file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class SiamOutputAsset(GenericAsset):
    """SIAM Output Data file asset."""

    regex_parse_str = r".+\.SiamOutput$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["siam-output-file", "ras-file", MediaType.TEXT]
        description = "SIAM Output Data file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class WaterQualityLogAsset(GenericAsset):
    """Water Quality Log file asset."""

    regex_parse_str = r".+\.bco$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["water-quality-log", "ras-file", MediaType.TEXT]
        description = "Water quality log file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class ColorScalesAsset(GenericAsset):
    """Color Scales file asset."""

    regex_parse_str = r".+\.color-scales$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["color-scales", "ras-file", MediaType.TEXT]
        description = "File that contains the water quality color scale."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class ComputationalMessageAsset(GenericAsset):
    """Computational Message file asset."""

    regex_parse_str = r".+\.comp-msgs.txt$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["computational-message-file", "ras-file", MediaType.TEXT]
        description = "Computational Message text file which contains messages from the computation process."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class UnsteadyRunFileAsset(GenericAsset):
    """Run file for Unsteady Flow asset."""

    regex_parse_str = r".+\.x\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["run-file", "ras-file", MediaType.TEXT]
        description = "Run file for Unsteady Flow simulations."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class OutputFileAsset(GenericAsset):
    """Output RAS file asset."""

    regex_parse_str = r".+\.o\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["output-file", "ras-file", MediaType.TEXT]
        description = "Output RAS file which contains all computed results."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class InitialConditionsFileAsset(GenericAsset):
    """Initial Conditions file asset."""

    regex_parse_str = r".+\.IC\.O\d{2}$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["initial-conditions-file", "ras-file", MediaType.TEXT]
        description = "Initial conditions file for unsteady flow plan."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class PlanRestartFileAsset(GenericAsset):
    """Restart file for Unsteady Flow Plan asset."""

    regex_parse_str = r".+\.p\d{2}\.rst$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["restart-file", "ras-file", MediaType.TEXT]
        description = "Restart file for unsteady flow plan."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class RasMapperFileAsset(GenericAsset):
    """RAS Mapper file asset."""

    regex_parse_str = r".+\.rasmap$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["ras-mapper-file", "ras-file", MediaType.TEXT]
        description = "RAS Mapper file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class RasMapperBackupFileAsset(GenericAsset):
    """Backup RAS Mapper file asset."""

    regex_parse_str = r".+\.rasmap\.backup$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["ras-mapper-file", "ras-file", MediaType.TEXT]
        description = "Backup RAS Mapper file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class RasMapperOriginalFileAsset(GenericAsset):
    """Original RAS Mapper file asset."""

    regex_parse_str = r".+\.rasmap\.original$"

    def __init__(self, href: str, *args, **kwargs):
        roles = ["ras-mapper-file", "ras-file", MediaType.TEXT]
        description = "Original RAS Mapper file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class MiscTextFileAsset(GenericAsset):
    """Miscellaneous Text file asset."""

    regex_parse_str = r".+\.txt$"

    def __init__(self, href: str, *args, **kwargs):
        roles = [MediaType.TEXT]
        description = "Miscellaneous text file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


class MiscXMLFileAsset(GenericAsset):
    """Miscellaneous XML file asset."""

    regex_parse_str = r".+\.xml$"

    def __init__(self, href: str, *args, **kwargs):
        roles = [MediaType.XML]
        description = "Miscellaneous XML file."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)


RAS_ASSET_CLASSES = [
    ProjectAsset,
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
