import datetime as dt
import json
import logging
import os
import io
from enum import Enum
import contextily as ctx
from functools import lru_cache
from typing import Any
import geopandas as gpd
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import dataretrieval.nwis as nwis
from pyproj import CRS, Transformer
from pystac import Collection, Item
from pystac.extensions.projection import ProjectionExtension
from shapely import Geometry, Polygon, simplify, to_geojson, union_all
from shapely.ops import transform

from ..utils.placeholders import (
    NULL_DATETIME,
    NULL_GEOMETRY,
    NULL_STAC_BBOX,
    NULL_STAC_GEOMETRY,
    PLACEHOLDER_ID,
)

from .const import SCHEMA_URI
from .ras_asset import (
    GeometryAsset,
    GeometryHdfAsset,
    PlanAsset,
    PlanHdfAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
    GenericAsset,
)
from .stac_utils import asset_factory


class ThumbnailParameter(Enum):
    XS = "cross_sections"
    MESH_2D = "mesh_areas"
    EXAMPLE = "example"


class RasModelItem(Item):
    def __init__(
        self,
        prj_file: str,
        crs: str = "",
        href: str = "",
        simplify_tolerance: float | None = 0.001,
        collection: str | Collection | None = None,
    ):
        id = PLACEHOLDER_ID
        properties = {}
        stac_extensions = [SCHEMA_URI]
        self._bbox = None
        self._geometry = None
        self._datetime = None
        self._start_datetime = None
        self._end_datetime = None
        super().__init__(
            id,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            properties,
            stac_extensions=stac_extensions,
            href=href,
            collection=collection,
        )
        self.prj_file = prj_file
        self.crs = crs
        self.href = href
        self._project: ProjectAsset | None = None
        self._plan_files: list[PlanAsset] = []
        self._flow_files: list[SteadyFlowAsset | UnsteadyFlowAsset | QuasiUnsteadyFlowAsset] = []
        self._geom_files: list[GeometryAsset | GeometryHdfAsset] = []
        self._files_with_associated_assets: list[ProjectAsset | PlanAsset | PlanHdfAsset | GeometryHdfAsset] = []
        self._has_1d = False
        self._has_2d = False
        self._dts: list[dt.datetime] = []
        self._datetime_source: str | None = None
        self.simplify_tolerance = simplify_tolerance

    def _transform_geometry(self, geom: Geometry) -> Geometry:
        pyproj_crs = CRS.from_user_input(self.crs)
        wgs_crs = CRS.from_authority("EPSG", "4326")
        if pyproj_crs != wgs_crs:
            transformer = Transformer.from_crs(pyproj_crs, wgs_crs, True)
            return transform(transformer.transform, geom)
        return geom

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if self._geometry == NULL_GEOMETRY:
            return NULL_STAC_BBOX
        return self._geometry.bounds

    @bbox.setter
    def bbox(self, stac_bbox: tuple[float, float, float, float]):
        if self._bbox == None:
            self._bbox = stac_bbox
        else:
            raise ValueError("Bounding box is already initialized and should not be set")

    @property
    def geometry(self) -> dict[str, Any]:
        """
        gets geometry using either list of geom assets or list of hdf assets, perhaps simplified to a given tolerance to reduce replication of data
        (ie item would record simplified geometry used when searching collection,
        asset would have more exact geometry representing contents of geom or hdf files)
        """
        # if geometry is equal to null placeholder, continue, else return current value
        if self._geometry == NULL_GEOMETRY:
            # if hdf file is present, get mesh areas from model, simplify, and use as geometry
            if self.has_2d:
                logging.info("Creating geometry using hdf file mesh areas")
                mesh_area_polygons: list[Polygon] = []
                for geom_asset in self.geometry_files:
                    if isinstance(geom_asset, GeometryHdfAsset):
                        if self.simplify_tolerance:
                            mesh_areas = simplify(
                                self._transform_geometry(geom_asset.mesh_areas),
                                self.simplify_tolerance,
                            )
                        else:
                            mesh_areas = self._transform_geometry(geom_asset.mesh_areas)
                        mesh_area_polygons.append(mesh_areas)
                self._geometry = union_all(mesh_area_polygons)
                stac_geom = json.loads(to_geojson(self._geometry))
                return stac_geom
            # if hdf file is not present, get concave hull of cross sections and use as geometry
            if self.has_1d:
                logging.info("Creating geometry using gNN file cross sections")
                concave_hull_polygons: list[Polygon] = []
                for geom_asset in self.geometry_files:
                    if isinstance(geom_asset, GeometryAsset):
                        if self.simplify_tolerance:
                            concave_hull = simplify(
                                self._transform_geometry(geom_asset.concave_hull),
                                self.simplify_tolerance,
                            )
                        else:
                            concave_hull = self._transform_geometry(geom_asset.concave_hull)
                        concave_hull_polygons.append(concave_hull)
                self._geometry = union_all(concave_hull_polygons)
                stac_geom = json.loads(to_geojson(self._geometry))
                return stac_geom
        stac_geom = json.loads(to_geojson(self._geometry))
        return stac_geom

    @geometry.setter
    def geometry(self, stac_geometry: dict) -> None:
        if self._geometry == None:
            if stac_geometry["coordinates"][0] == []:
                self._geometry = Polygon()
            else:
                self._geometry = Polygon(stac_geometry["coordinates"][0])
        else:
            raise ValueError("Geometry is already initialized and should not be set")

    @property
    @lru_cache
    def datetime(self) -> dt.datetime:
        if len(self._dts) == 0:
            self._datetime = dt.datetime.now()
            return self._datetime
        if len(self._dts) == 1:
            self._datetime = self._dts[0]
        # if has start and end, define datetime as start
        self._datetime = self.start_datetime
        return self._datetime

    @datetime.setter
    def datetime(self, stac_datetime: dt.datetime | None) -> None:
        if self._datetime == None:
            self._datetime = stac_datetime
        else:
            raise ValueError("Datetime is already initialized and should not be set")

    @property
    @lru_cache
    def start_datetime(self) -> dt.datetime | None:
        if len(self._dts) <= 1:
            self._start_datetime = None
            return self._start_datetime
        min_dt = min(self._dts)
        max_dt = max(self._dts)
        if min_dt == max_dt:
            self._start_datetime = None
            return self._start_datetime
        self._start_datetime = min_dt
        return self._start_datetime

    @start_datetime.setter
    def start_datetime(self, stac_start_datetime: dt.datetime | None) -> None:
        if self._start_datetime == None:
            self._start_datetime = stac_start_datetime
        else:
            raise ValueError("Start datetime is already initialized and should not be set")

    @property
    @lru_cache
    def end_datetime(self) -> dt.datetime | None:
        if len(self._dts) <= 1:
            self._end_datetime = None
            return self._end_datetime
        min_dt = min(self._dts)
        max_dt = max(self._dts)
        if min_dt == max_dt:
            self._end_datetime = None
            return self._end_datetime
        self._end_datetime = max_dt
        return self._end_datetime

    @end_datetime.setter
    def end_datetime(self, stac_end_datetime: dt.datetime | None) -> None:
        if self._end_datetime == None:
            self._end_datetime = stac_end_datetime
        else:
            raise ValueError("End datetime is already initialized and should not be set")

    def populate(self) -> None:
        # searches directory of prj file for files to parse as assets associated with project, then adds these as assets, storing the project asset as self.project when found
        parent = os.path.dirname(self.prj_file)
        for entry in os.scandir(parent):
            if entry.is_file():
                self.add_asset(entry.path)
        # explicitly set internal STAC properties dictionary
        self.properties["ras:datetime_source"] = self.datetime_source
        self.properties["ras:has_1d"] = self.has_1d
        self.properties["ras:has_2d"] = self.has_2d
        # once all assets are created, populate associations between assets
        for asset in self._files_with_associated_assets:
            asset.associate_related_assets(self.assets)
        self.validate()

    def add_asset(self, url: str) -> None:
        """Add an asset to the item."""
        # this is where properties derived from singular asset content (self.has_1d, self.has_2d, self._dts) might be populated
        asset = asset_factory(url, self.crs)
        super().add_asset(asset.title, asset)
        if isinstance(asset, ProjectAsset):
            if self._project is not None:
                f"Only one project asset is allowed. Found {str(asset)} when {str(self._project)} was already set."
            self._project = asset
            self._files_with_associated_assets.append(asset)
        elif isinstance(asset, (PlanAsset, PlanHdfAsset)):
            self._plan_files.append(asset)
            self._files_with_associated_assets.append(asset)
        elif isinstance(asset, (GeometryAsset, GeometryHdfAsset)):
            self.ensure_projection_schema()
            self._geom_files.append(asset)
            if isinstance(asset, GeometryHdfAsset):
                self._files_with_associated_assets.append(asset)
                if self._has_1d == False and asset.cross_sections != None and asset.cross_sections > 0:
                    self._has_1d = True
                if self._has_2d == False and asset.mesh_areas != None:
                    self._has_2d = True
            if isinstance(asset, GeometryAsset):
                if self._has_1d == False and asset.has_1d:
                    self._has_1d = True
                if self._has_2d == False and asset.has_2d:
                    self._has_2d = True
                self._dts.extend(asset.datetimes)
        elif isinstance(asset, (SteadyFlowAsset, QuasiUnsteadyFlowAsset, UnsteadyFlowAsset)):
            self._flow_files.append(asset)

    def ensure_projection_schema(self) -> None:
        ProjectionExtension.ensure_has_extension(self, True)

    @property
    @lru_cache
    def project_file(self) -> ProjectAsset:
        # checks if project has been set and if not, autofinds project assets
        if self._project == None:
            self.populate()
        # if project is still None after autofinding, raise exception
        if self._project == None:
            raise FileNotFoundError(f"no project file found")
        # after autofind has been run (building all assets associated with the item in the progress), create links between project asset and assets it references
        self._project.associate_related_assets(self.assets)
        return self._project

    @property
    def plan_files(self) -> list[PlanAsset | PlanHdfAsset]:
        if self._project == None:
            self.populate()
        return self._plan_files

    @property
    def geometry_files(self) -> list[GeometryAsset | GeometryHdfAsset]:
        if self._project == None:
            self.populate()
        return self._geom_files

    @property
    def flow_files(
        self,
    ) -> list[SteadyFlowAsset | QuasiUnsteadyFlowAsset | UnsteadyFlowAsset]:
        if self._project == None:
            self.populate()
        return self._flow_files

    @property
    def datetime_source(self) -> str:
        if self._datetime_source == None:
            if self._dts == None:
                self.populate()
            if len(self._dts) == 0:
                self._datetime_source = "processing_time"
            else:
                self._datetime_source = "model_geometry"
        return self._datetime_source

    @property
    def has_1d(self) -> bool:
        return self._has_1d

    @property
    def has_2d(self) -> bool:
        return self._has_2d

    def _plot_mesh_areas(self, ax, mesh_polygons: gpd.GeoDataFrame) -> list:
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

    def _plot_breaklines(self, ax, breaklines: gpd.GeoDataFrame) -> list:
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

    def _plot_bc_lines(self, ax, bc_lines: gpd.GeoDataFrame) -> list:
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

    def _plot_usgs_gages(self, ax, usgs_gages: gpd.GeoDataFrame) -> list:
        legend_handles = [Line2D([0], [0], color="none", linestyle="None", label="USGS Gages")]
        gage_colors = plt.cm.get_cmap("Set1", len(usgs_gages))

        for gage, color in zip(usgs_gages.itertuples(), gage_colors.colors):
            ax.plot(
                gage.geometry.x,
                gage.geometry.y,
                color=color,
                marker="*",
                markersize=15,
                label=gage.site_no,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="*",
                    markersize=15,
                    linestyle="None",
                    label=gage.site_no,
                )
            )
        return legend_handles

    def add_thumbnail_asset(self, filepath: str) -> None:
        if filepath.startswith("s3://"):
            media_type = "image/png"
        else:
            # Ensure the file exists before adding as an asset
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Thumbnail file not found: {filepath}")

            media_type = "image/png"

        self.assets["thumbnail"] = GenericAsset(
            href=filepath,
            title="Model Thumbnail",
            description="Thumbnail image for the model",
            media_type=media_type,
            roles=["thumbnail"],
            extra_fields=None,
        )

    def get_primary_geom(self):

        geom_hdf_assets = [asset for asset in self._geom_files if isinstance(asset, GeometryHdfAsset)]
        if len(geom_hdf_assets) == 0:
            raise FileNotFoundError("No 2D geometry found")
        elif len(geom_hdf_assets) > 1:
            primary_geom_hdf_asset = next(asset for asset in geom_hdf_assets if ".g01" in asset.hdf_file)
        else:
            primary_geom_hdf_asset = geom_hdf_assets[0]
        return primary_geom_hdf_asset

    def thumbnail(
        self,
        add_asset: bool,
        write: bool,
        parameters: list,
        title: str = "Model Thumbnail",
    ) -> Figure:
        # create thumbnail figure
        # if add_asset or write is true, save asset to filepath relative to item href and add thumbnail asset to asset dict
        if self._has_2d:
            primary_geom_hdf_asset = self.get_primary_geom()

            fig, ax = plt.subplots(figsize=(12, 12))
            legend_handles = []

            for parameter in parameters:
                if parameter == "usgs_gages":
                    gages_gdf = self.get_usgs_data(False)
                    gages_gdf_geo = gages_gdf.to_crs(self.crs)
                    legend_handles += self._plot_usgs_gages(ax, gages_gdf_geo)
                else:
                    if not hasattr(primary_geom_hdf_asset, parameter):
                        raise AttributeError(f"Parameter {parameter} not found in {primary_geom_hdf_asset.hdf_file}")

                    if parameter == "mesh_areas":
                        parameter_data = primary_geom_hdf_asset.mesh_areas(return_gdf=True)
                    else:
                        parameter_data = getattr(primary_geom_hdf_asset, parameter)
                    parameter_data_geo = parameter_data.to_crs(self.crs)

                    if parameter == "mesh_areas":
                        legend_handles += self._plot_mesh_areas(ax, parameter_data_geo)
                    elif parameter == "breaklines":
                        legend_handles += self._plot_breaklines(ax, parameter_data_geo)
                    elif parameter == "bc_lines":
                        legend_handles += self._plot_bc_lines(ax, parameter_data_geo)

            # Add OpenStreetMap basemap
            ctx.add_basemap(
                ax,
                crs=f"EPSG:{self.crs}",
                source=ctx.providers.OpenStreetMap.Mapnik,
                alpha=0.4,
            )
            ax.set_title(title, fontsize=15)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1, 0.5))
            if add_asset or write:
                filepath = self.href.rsplit("/", 1)[0] + "/thumbnail.png"
                if filepath.startswith("s3://"):
                    img_data = io.BytesIO()
                    fig.savefig(img_data, format="png", bbox_inches="tight")
                    img_data.seek(0)
                    save_bytes_s3(img_data, filepath)

                else:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    fig.savefig(filepath, dpi=80, bbox_inches="tight")

                if add_asset:
                    self.add_thumbnail_asset(filepath)

    def add_usgs_properties(self, usgs_gages: gpd.GeoDataFrame) -> None:
        """
        Adds USGS metadata to the STAC item properties.
        """

        usgs_metadata_list = []
        for _, gage in usgs_gages.iterrows():
            gage_metadata = {
                "site_no": gage["site_no"],
                "name": gage["station_nm"],
                "latitude": gage["dec_lat_va"],
                "longitude": gage["dec_long_va"],
                "associated_ref_line": gage["refln_name"],
            }
            usgs_metadata_list.append(gage_metadata)

        self.properties["usgs_gages"] = usgs_metadata_list

    def get_usgs_data(
        self,
        add_properties: bool,
        buffer_increase=0.0001,
        max_buffer=0.01,
    ) -> gpd.GeoDataFrame:
        # retrieve USGS gages using model reference lines from HDF asset, if available, else raise exception
        # if add_properties is true, create USGS metadata JSON item for each gage and add it to array property
        primary_geom_hdf_asset = self.get_primary_geom()
        ref_line = primary_geom_hdf_asset.reference_lines
        ref_line = ref_line.to_crs(self.crs)
        all_usgs_gages = pd.DataFrame()

        for _, row in ref_line.iterrows():
            buffer_increment = 0  # Start with no buffer

            while True:
                try:
                    if buffer_increment == 0:
                        bbox = [*row.geometry.bounds]
                    else:
                        bbox = [*row.geometry.buffer(buffer_increment).bounds]

                    # bbox must be rounded to work with usgs data retrieval
                    rounded_bbox = [round(coord, 7) for coord in bbox]

                    usgs_data = nwis.what_sites(bBox=rounded_bbox)

                    usgs_gages = usgs_data[0]
                    # Filter out any gages where 'site_no' has more than 8 digits, assuming those are invalid
                    valid_gages = usgs_gages[usgs_gages["site_no"].str.len() == 8]
                    valid_gages["refln_name"] = row.refln_name

                    # If there are valid gages, append them to all_usgs_gages and break the loop
                    if not valid_gages.empty:
                        logging.debug(f"Found gage for reference line {row.refln_name}.")
                        all_usgs_gages = pd.concat([all_usgs_gages, valid_gages], ignore_index=True)
                        break

                    logging.debug(
                        f"No valid gages found. Increasing buffer by {buffer_increase} for reference line {row.refln_name}."
                    )
                    buffer_increment += buffer_increase

                    if buffer_increment > max_buffer:
                        logging.debug(
                            f"Max buffer of {max_buffer} reached. No gages found for reference line {row.refln_name}."
                        )
                        break
                except ValueError:
                    logging.debug(
                        f"No gages found. Increasing buffer by {buffer_increase} for reference line {row.refln_name}."
                    )
                    buffer_increment += buffer_increase

                    if buffer_increment > max_buffer:
                        logging.warning(
                            f"Max buffer of {max_buffer} reached. No gages found for reference line {row.refln_name}."
                        )
                        break

        # Remove duplicate gages
        all_usgs_gages = all_usgs_gages.drop_duplicates(subset="site_no").reset_index(drop=True)
        logging.info(f"Found {len(all_usgs_gages)} unique gages.")

        if add_properties:
            self.add_usgs_properties(all_usgs_gages)
        return all_usgs_gages
