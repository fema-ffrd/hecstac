import datetime as dt
import json
import os
from enum import Enum
from functools import lru_cache
from typing import Any

from matplotlib.figure import Figure
from pyproj import CRS, Transformer
from pystac import Collection, Item
from shapely import Geometry, Polygon, simplify, to_geojson, union_all
from shapely.ops import transform

from ..utils.placeholders import (
    NULL_DATETIME,
    NULL_GEOMETRY,
    NULL_STAC_BBOX,
    NULL_STAC_GEOMETRY,
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
        simplify_tolerance: float | None = 0.01,
        collection: str | Collection | None = None,
    ):
        id = ""
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
        self._project: ProjectAsset | None = None
        self._plan_files: list[PlanAsset] = []
        self._flow_files: list[SteadyFlowAsset | UnsteadyFlowAsset | QuasiUnsteadyFlowAsset] = []
        self._geom_files: list[GeometryAsset | GeometryHdfAsset] = []
        self._linkable_files: list[ProjectAsset | PlanAsset | PlanHdfAsset | GeometryHdfAsset] = []
        self._has_1d = False
        self._has_2d = False
        self._dts: list[dt.datetime] = []
        self._datetime_source: str | None = None
        self.simplify_tolerance = simplify_tolerance

    def _transform_geometry(self, geom: Geometry) -> Geometry:
        pyproj_crs = CRS.from_user_input(self.crs)
        wgs_crs = CRS.from_authority("EPSG", "4326")
        if pyproj_crs != wgs_crs:
            transformer = Transformer.from_crs(pyproj_crs, wgs_crs)
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
                mesh_area_polygons: list[Polygon] = []
                for geom_asset in self.geometry_files:
                    if isinstance(geom_asset, GeometryHdfAsset):
                        if self.simplify_tolerance:
                            mesh_areas = simplify(geom_asset.mesh_areas, self.simplify_tolerance)
                        else:
                            mesh_areas = geom_asset.mesh_areas
                        mesh_area_polygons.append(mesh_areas)
                unioned_geometry = union_all(mesh_area_polygons)
                self._geometry = self._transform_geometry(unioned_geometry)
                stac_geom = json.loads(to_geojson(self._geometry))
                return stac_geom
            # if hdf file is not present, get concave hull of cross sections and use as geometry
            if self.has_1d:
                concave_hull_polygons: list[Polygon] = []
                for geom_asset in self.geometry_files:
                    if isinstance(geom_asset, GeometryAsset):
                        if self.simplify_tolerance:
                            concave_hull = simplify(geom_asset.concave_hull, self.simplify_tolerance)
                        else:
                            concave_hull = geom_asset.concave_hull
                        concave_hull_polygons.append(concave_hull)
                unioned_geometry = union_all(concave_hull_polygons)
                self._geometry = self._transform_geometry(unioned_geometry)
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
    def datetime(self) -> str:
        if len(self._dts) == 0:
            self._datetime_source = "processing_time"
            self._datetime = dt.datetime.now().isoformat()
            return self._datetime
        self._datetime_source = "model_geometry"
        if len(self._dts) == 1:
            self._datetime = self._dts[0].isoformat()
        # if has start and end, define datetime as start
        self._datetime = self._start_datetime
        return self._datetime

    @datetime.setter
    def datetime(self, stac_datetime: dt.datetime | None) -> None:
        if self._datetime == None:
            self._datetime = stac_datetime
        else:
            raise ValueError("Datetime is already initialized and should not be set")

    @property
    @lru_cache
    def start_datetime(self) -> str | None:
        if len(self._dts) <= 1:
            self._start_datetime = None
            return self._start_datetime
        min_dt = min(self._dts)
        max_dt = max(self._dts)
        if min_dt == max_dt:
            self._start_datetime = None
            return self._start_datetime
        self._start_datetime = min_dt
        return self._start_datetime.isoformat()

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
        return self._end_datetime.isoformat()

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
            self.add_asset(entry.path)
        # explicitly set internal STAC properties dictionary
        self.properties["datetime"] = self.datetime
        self.properties["start_datetime"] = self.start_datetime
        self.properties["end_datetime"] = self.end_datetime
        self.extra_fields["ras:datetime_source"] = self._datetime_source
        self.extra_fields["ras:has_1d"] = self.has_1d
        self.extra_fields["ras:has_2d"] = self.has_2d
        # once all assets are created, populate links between assets
        for asset in self._linkable_files:
            print(f"creating link for asset {asset}")
            asset.create_links(self.assets)

    def add_asset(self, url: str) -> None:
        """Add an asset to the item."""
        # this is where properties derived from singular asset content (self.has_1d, self.has_2d, self._dts) might be populated
        asset = asset_factory(url, self.crs)
        super().add_asset(asset.title, asset)
        if isinstance(asset, ProjectAsset):
            if self._project is not None:
                f"Only one project asset is allowed. Found {str(asset)} when {str(self._project)} was already set."
            self._project = asset
            self._linkable_files.append(asset)
        elif isinstance(asset, (PlanAsset, PlanHdfAsset)):
            self._plan_files.append(asset)
            self._linkable_files.append(asset)
        elif isinstance(asset, (GeometryAsset, GeometryHdfAsset)):
            self._geom_files.append(asset)
            if isinstance(asset, GeometryHdfAsset):
                self._linkable_files.append(asset)
                # TODO: figure out how to determine has_1d, has_2d, and _dts from HDF asset alone
            if isinstance(asset, GeometryAsset):
                if self._has_1d == False and asset.has_1d:
                    self._has_1d = True
                if self._has_2d == False and asset.has_2d:
                    self._has_2d = True
                self._dts.extend(asset.datetimes)
        elif isinstance(asset, (SteadyFlowAsset, QuasiUnsteadyFlowAsset, UnsteadyFlowAsset)):
            self._flow_files.append(asset)

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
        self._project.create_links(self.assets)
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
    def flow_files(self) -> list[SteadyFlowAsset | QuasiUnsteadyFlowAsset | UnsteadyFlowAsset]:
        if self._project == None:
            self.populate()
        return self._flow_files

    @property
    def datetime_source(self) -> str:
        return self._datetime_source

    @property
    def has_1d(self) -> bool:
        return self._has_1d

    @property
    def has_2d(self) -> bool:
        return self._has_2d

    def thumbnail(
        self, add_asset: bool, write: bool, parameter: ThumbnailParameter = ThumbnailParameter.MESH_2D
    ) -> Figure:
        # create thumbnail figure
        # if add_asset or write is true, save asset to filepath relative to item href and add thumbnail asset to asset dict
        pass

    # def get_usgs_data(self, add_properties: bool) -> GeoDataFrame:
    #     # retrieve USGS gages using model reference lines from HDF asset, if available, else raise exception
    #     # if add_properties is true, create USGS metadata JSON item for each gage and add it to array property
    #     pass
