import datetime
import datetime as dt
import io
import json
import logging
import os
from enum import Enum
from functools import lru_cache
from typing import Any

import contextily as ctx
import dataretrieval.nwis as nwis
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pyproj import CRS, Transformer
from pystac import Collection, Item
from pystac.extensions.projection import ProjectionExtension
from shapely import Geometry, Polygon, box, simplify, to_geojson, union_all
from shapely.ops import transform

NULL_DATETIME = datetime.datetime(9999, 9, 9)
NULL_GEOMETRY = Polygon()
NULL_STAC_GEOMETRY = json.loads(to_geojson(NULL_GEOMETRY))
NULL_BBOX = box(0, 0, 0, 0)
NULL_STAC_BBOX = NULL_BBOX.bounds
PLACEHOLDER_ID = "id"

from hecstac.common.asset_factory import AssetFactory

from .assets import (
    GenericAsset,
    GeometryAsset,
    GeometryHdfAsset,
    PlanAsset,
    PlanHdfAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
    RAS_EXTENSION_MAPPING
)
from .consts import SCHEMA_URI
from .stac_utils import asset_factory


class ThumbnailParameter(Enum):
    XS = "cross_sections"
    MESH_2D = "mesh_areas"
    EXAMPLE = "example"


class RasModelItem(Item):
    def __init__(
        self,
        prj_file: str,
        crs: str,
        item_id: str = "",
        href: str = "",
        simplify_tolerance: float | None = 0.001,
        collection: str | Collection | None = None,
    ):
        if not item_id:
            item_id = PLACEHOLDER_ID
        properties = {}
        stac_extensions = [SCHEMA_URI]
        self._bbox = None
        self._geometry = None
        self._datetime = None
        self._start_datetime = None
        self._end_datetime = None
        super().__init__(
            item_id,
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
        self.factory = AssetFactory(RAS_EXTENSION_MAPPING)

    def _geometry_to_wgs84(self, geom: Geometry) -> Geometry:
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
                                self._geometry_to_wgs84(geom_asset.mesh_areas(self.crs)),
                                self.simplify_tolerance,
                            )
                        else:
                            mesh_areas = self._geometry_to_wgs84(geom_asset.mesh_areas(self.crs))
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
                                self._geometry_to_wgs84(geom_asset.concave_hull),
                                self.simplify_tolerance,
                            )
                        else:
                            concave_hull = self._geometry_to_wgs84(geom_asset.concave_hull)
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
        """Searches directory of prj file for files to parse as assets associated with project,
        then adds these as assets, storing the project asset as self.project when found."""

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
        """Add an asset to the item and categorize it based on its type."""

        asset = self.factory.create_asset(url)
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

    def add_model_thumbnail(self, add_asset: bool, write: bool, layers: list, title: str = "Model_Thumbnail"):

        for geom in self._geom_files:
            if isinstance(geom, GeometryHdfAsset):
                if add_asset:
                    self.assets["thumbnail"] = geom.thumbnail(
                        add_asset=add_asset, write=write, layers=layers, title=title, thumbnail_dest=self.href
                    )
