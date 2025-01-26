import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer
from pystac import Item
from pystac.extensions.projection import ProjectionExtension
from shapely import Geometry, Polygon, box, simplify, to_geojson, union_all
from shapely.ops import transform

from hecstac.common.path_manager import LocalPathManager
from hecstac.ras.parser import ProjectFile

NULL_DATETIME = datetime.datetime(9999, 9, 9)
NULL_GEOMETRY = Polygon()
NULL_STAC_GEOMETRY = json.loads(to_geojson(NULL_GEOMETRY))
NULL_BBOX = box(0, 0, 0, 0)
NULL_STAC_BBOX = NULL_BBOX.bounds
PLACEHOLDER_ID = "id"

from hecstac.common.asset_factory import AssetFactory
from hecstac.ras.assets import (
    RAS_EXTENSION_MAPPING,
    GeometryAsset,
    GeometryHdfAsset,
    PlanAsset,
    PlanHdfAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
)

# from hecstac.ras.consts import SCHEMA_URI


# class ThumbnailParameter(Enum):
#     XS = "cross_sections"
#     MESH_2D = "mesh_areas"
#     EXAMPLE = "example"


class RASModelItem(Item):
    """An object representation of a HEC-RAS model."""

    PROJECT = "ras:project"
    PROJECT_TITLE = "ras:project_title"
    MODEL_UNITS = "ras:unit system"
    MODEL_GAGES = "ras:gages"
    PROJECT_VERSION = "ras:version"
    PROJECT_DESCRIPTION = "ras:description"
    PROJECT_STATUS = "ras:status"
    PROJECT_UNITS = "ras:unit_system"

    RAS_HAS_1D = "ras:has_1d"
    RAS_HAS_2D = "ras:has_2d"
    RAS_DATETIME_SOURCE = "ras:datetime_source"

    def __init__(self, ras_project_file, item_id: str, simplify_geometry: bool = True):

        self._project = None
        self.assets = {}
        self.links = []
        self.thumbnail_paths = []
        self.geojson_paths = []
        self.extra_fields = {}
        self.stac_extensions = None
        self.pm = LocalPathManager(Path(ras_project_file).parent)
        self._href = self.pm.item_path(item_id)
        self.ras_project_file = ras_project_file
        self._simplify_geometry = simplify_geometry

        self.pf = ProjectFile(self.ras_project_file)
        self.factory = AssetFactory(RAS_EXTENSION_MAPPING)

        super().__init__(
            Path(self.ras_project_file).stem,
            self._geometry,
            self._bbox,
            self._datetime,
            self._properties,
            href=self._href,
        )

    @property
    def _properties(self) -> None:
        """Properties for the RAS STAC item."""
        properties = {}

        properties = {}
        properties[self.PROJECT_TITLE] = self.pf.project_title
        properties[self.PROJECT_VERSION] = self.pf.ras_version
        properties[self.PROJECT_DESCRIPTION] = self.pf.project_description
        properties[self.PROJECT_STATUS] = self.pf.project_status
        properties[self.MODEL_UNITS] = self.pf.project_units

        # self.properties[RAS_DATETIME_SOURCE] = self.datetime_source
        # self.properties[RAS_HAS_1D] = self.has_1d
        # self.properties[RAS_HAS_2D] = self.has_2d
        # once all assets are created, populate associations between assets
        return properties

    @property
    def _bbox(self) -> tuple[float, float, float, float]:
        if self._geometry == NULL_STAC_GEOMETRY:
            return NULL_STAC_BBOX
        return self._geometry.bounds

    @property
    def _geometry(self) -> dict | None:
        """
        gets geometry using either list of geom assets or list of hdf assets, perhaps simplified to
        a given tolerance to reduce replication of data
        (ie item would record simplified geometry used when searching collection,
        asset would have more exact geometry representing contents of geom or hdf files)
        """
        # if geometry is equal to null placeholder, continue, else return current value
        geometries = []
        if 2 == 3:
            # if self.has_2d:
            geometries.append(self.parse_2d_geom())

        # if hdf file is not present, get concave hull of cross sections and use as geometry
        # if self.has_1d:
        if 1 == 2:
            geometries.append(self.parse_1d_geom())

        if len(geometries) == 0:
            logging.error("No geometry found for RAS item.")
            return NULL_STAC_GEOMETRY

        unioned_geometry = union_all(geometries)
        if self._simplify_geometry:
            unioned_geometry = simplify(unioned_geometry, self.simplify_tolerance)

        return json.loads(to_geojson(unioned_geometry))

    @property
    def _datetime(self) -> datetime:
        """The datetime for the HMS STAC item."""
        # date = datetime.strptime(self.pf.basins[0].header.attrs["Last Modified Date"], "%d %B %Y")
        # time = datetime.strptime(self.pf.basins[0].header.attrs["Last Modified Time"], "%H:%M:%S").time()
        return datetime.datetime.now()

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

    def add_model_thumbnail(self, add_asset: bool, write: bool, layers: list, title: str = "Model_Thumbnail"):

        for geom in self._geom_files:
            if isinstance(geom, GeometryHdfAsset):
                if add_asset:
                    self.assets["thumbnail"] = geom.thumbnail(
                        add_asset=add_asset, write=write, layers=layers, title=title, thumbnail_dest=self.href
                    )

    def add_ras_asset(self, fpath: str = None) -> None:
        """Add an asset to the HMS STAC item."""
        # fpath = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.prj"
        # fpath = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.p01"
        # fpath = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.g01"
        # fpath = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.u01"
        fpath = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.p01.hdf"
        # fpath = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.g01.hdf"

        if os.path.exists(fpath):
            asset = self.factory.create_ras_asset(fpath)
            logging.info(f"Adding asset {str(asset)}")
            if asset is not None:
                self.add_asset(asset.title, asset)
                if isinstance(asset, ProjectAsset):
                    if self._project is not None:
                        logging.error(
                            f"Only one project asset is allowed. Found {str(asset)} when {str(self._project)} was already set."
                        )
                    self._project = asset

    def parse_1d_geom(self):
        logging.info("Creating geometry using 1d text file cross sections")
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
        return self._geometry

    def parse_2d_geom(self):
        logging.info("Creating 2D geometry elements using hdf file mesh areas")
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
        return union_all(mesh_area_polygons)

    def ensure_projection_schema(self) -> None:
        ProjectionExtension.ensure_has_extension(self, True)

    def scan_model_dir(self):
        base_dir = os.path.dirname(self.ras_project_file)
        files = []
        for root, _, filenames in os.walk(base_dir):
            depth = root[len(base_dir) :].count(os.sep)
            if depth > 1:
                break
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files
