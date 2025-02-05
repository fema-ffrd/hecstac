import datetime
import json
import logging
import os
from collections import UserDict
from functools import lru_cache
from pathlib import Path

import pystac
import pystac.errors
from pyproj import CRS, Transformer
from pystac import Item
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.storage import StorageExtension
from rashdf import RasGeomHdf
from shapely import Geometry, Polygon, simplify, to_geojson, union_all
from shapely.geometry import shape
from shapely.ops import transform

from hecstac.common.asset_factory import AssetFactory
from hecstac.common.geometry import reproject_to_wgs84
from hecstac.common.path_manager import LocalPathManager
from hecstac.ras.assets import (
    RAS_EXTENSION_MAPPING,
    GeometryAsset,
    GeometryHdfAsset,
    ProjectAsset,
)
from hecstac.ras.consts import (
    NULL_DATETIME,
    NULL_STAC_BBOX,
    NULL_STAC_GEOMETRY,
)
from hecstac.ras.parser import ProjectFile


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

    def __init__(self, *args, **kwargs):
        """Add a few default properties to the base class."""
        super().__init__(*args, **kwargs)
        self.simplify_geometry = True

    @classmethod
    def from_prj(cls, ras_project_file, item_id: str, crs: str = None, simplify_geometry: bool = True):
        """Create an item from a RAS .prj file."""
        pm = LocalPathManager(Path(ras_project_file).parent)
        href = pm.item_path(item_id)

        stac = cls(
            Path(ras_project_file).stem,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            {"ras_project_file": ras_project_file},
            href=href,
        )
        stac.crs
        if crs:
            stac.crs = crs
        stac.simplify_geometry = simplify_geometry

        for fpath in stac.scan_model_dir():
            if fpath and fpath != href:
                logging.info(f"Processing asset: {fpath}")
                stac.add_ras_asset(fpath)
        return stac

    @property
    def ras_project_file(self) -> str:
        """Get the path to the HEC-RAS .prj file."""
        return self._properties.get("ras_project_file")

    @property
    @lru_cache
    def factory(self) -> AssetFactory:
        """Return AssetFactory for this item."""
        return AssetFactory(RAS_EXTENSION_MAPPING)

    @property
    @lru_cache
    def pf(self) -> ProjectFile:
        """Get a ProjectFile instance for the RAS Model .prj file."""
        return ProjectFile(self.ras_project_file)

    @property
    def has_2d(self) -> bool:
        """Whether any geometry file has 2D elements."""
        return any([a.check_2d() for a in self.geometry_assets])

    @property
    def has_1d(self) -> bool:
        """Whether any geometry file has 2D elements."""
        return any([a.check_1d() for a in self.geometry_assets])

    @property
    def geometry_assets(self) -> list[RasGeomHdf]:
        """Return any RasGeomHdf in assets."""
        return [a for a in self.assets.values() if isinstance(a, RasGeomHdf)]

    @property
    def crs(self) -> CRS:
        """Get the authority code for the model CRS."""
        try:
            return CRS(self.ext.proj.wkt2)
        except pystac.errors.ExtensionNotImplemented:
            return None

    @crs.setter
    def crs(self, crs):
        """Apply the projection extension to this item given a CRS."""
        prj_ext = ProjectionExtension.ext(self, add_if_missing=True)
        crs = CRS(crs)
        prj_ext.apply(epsg=crs.to_epsg(), wkt2=crs.to_wkt())

    @property
    def geometry(self) -> dict:
        """Return footprint of model as a geojson."""
        if self.crs is None:
            logging.warning("Geometry requested for model with no spatial reference.")
            return NULL_STAC_GEOMETRY

        geometries = []

        if self.has_2d:
            geometries.append(self.parse_2d_geom())

        # if self.has_1d:
        #     geometries.append(self.parse_1d_geom())

        if len(geometries) == 0:
            logging.error("No geometry found for RAS item.")
            return NULL_STAC_GEOMETRY

        unioned_geometry = union_all(geometries)
        if self.simplify_geometry:
            unioned_geometry = simplify(unioned_geometry, 0.001)

        unioned_geometry = reproject_to_wgs84(unioned_geometry, self.crs)
        return json.loads(to_geojson(unioned_geometry))

    @property
    def bbox(self) -> list[float]:
        """Get the bounding box of the model geometry."""
        return shape(self.geometry).bounds

    @property
    def properties(self) -> None:
        """Properties for the RAS STAC item."""
        if self.ras_project_file is None:
            return self._properties
        properties = self._properties
        # properties[self.RAS_HAS_1D] = self.has_1d
        properties[self.RAS_HAS_2D] = self.has_2d
        properties[self.PROJECT_TITLE] = self.pf.project_title
        properties[self.PROJECT_VERSION] = self.pf.ras_version
        properties[self.PROJECT_DESCRIPTION] = self.pf.project_description
        properties[self.PROJECT_STATUS] = self.pf.project_status
        properties[self.MODEL_UNITS] = self.pf.project_units

        # TODO: once all assets are created, populate associations between assets
        return properties

    @properties.setter
    def properties(self, properties: dict):
        """Set properties."""
        self._properties = properties

    @property
    def datetime(self) -> datetime:
        """The datetime for the RAS STAC item."""
        item_datetime = None

        for geom_file in self.geometry_assets:
            if isinstance(geom_file, GeometryHdfAsset):
                geom_date = geom_file.hdf_object.geometry_time
                if geom_date:
                    item_datetime = geom_date
                    self.properties[self.RAS_DATETIME_SOURCE] = "model_geometry"
                    logging.info(f"Using item datetime from {geom_file.href}")
                    break

        if item_datetime is None:
            logging.warning("Could not extract item datetime from geometry, using item processing time.")
            item_datetime = datetime.datetime.now()
            self.properties[self.RAS_DATETIME_SOURCE] = "processing_time"
        return item_datetime

    def add_model_thumbnails(self, layers: list, title_prefix: str = "Model_Thumbnail", thumbnail_dir=None):
        """Generates model thumbnail asset for each geometry file.

        Parameters
        ----------
        layers : list
            List of geometry layers to be included in the plot. Options include 'mesh_areas', 'breaklines', 'bc_lines'
        title_prefix : str, optional
            Thumbnail title prefix, by default "Model_Thumbnail".
        """
        if thumbnail_dir:
            thumbnail_dest = thumbnail_dir
        else:
            thumbnail_dest = self.self_href

        for geom in self.geometry_assets:
            if isinstance(geom, GeometryHdfAsset):
                self.assets[f"{geom.href}_thumbnail"] = geom.thumbnail(
                    layers=layers, title=title_prefix, thumbnail_dest=thumbnail_dest
                )

        # TODO: Add 1d model thumbnails

    def add_ras_asset(self, fpath: str = "") -> None:
        """Add an asset to the RAS STAC item."""
        if not os.path.exists(fpath):
            logging.warning(f"File not found: {fpath}")
            return
        try:
            asset = self.factory.create_ras_asset(fpath)
            self.add_asset(asset.title, asset)
            # TODO:  add feature to set crs from geom hdf
            logging.debug(f"Adding asset {str(asset)}")
        except TypeError as e:
            logging.error(f"Error creating asset for {fpath}: {e}")

    def parse_1d_geom(self):
        """Read 1d geometry from concave hull."""
        logging.info("Creating geometry using 1d text file cross sections")
        concave_hull_polygons: list[Polygon] = []
        for geom_asset in self._geom_files:
            if isinstance(geom_asset, GeometryAsset):
                try:
                    geom_asset.crs = self.crs
                    concave_hull = geom_asset.geomf.concave_hull
                    concave_hull = self._geometry_to_wgs84(concave_hull)
                    concave_hull_polygons.append(concave_hull)
                except ValueError:
                    logging.warning(f"Could not extract geometry from {geom_asset.href}")

        return union_all(concave_hull_polygons)

    def parse_2d_geom(self):
        """Read 2d geometry from hdf file mesh areas."""
        mesh_area_polygons: list[Polygon] = []
        for geom_asset in self._geom_files:
            if isinstance(geom_asset, GeometryHdfAsset):
                logging.info(f"Extracting geom from mesh areas in {geom_asset.href}")

                mesh_areas = self._geometry_to_wgs84(geom_asset.hdf_object.mesh_areas(self.crs))
                mesh_area_polygons.append(mesh_areas)

        return union_all(mesh_area_polygons)

    def scan_model_dir(self):
        """Find all files in the project folder."""
        parent = Path(self.ras_project_file).parent
        stem = Path(self.ras_project_file).name.split(".")[0]
        return [str(i) for i in parent.glob(f"{stem}*")]

    ### Some properties are dynamically generated.  Ignore external updates ###

    @geometry.setter
    def geometry(self, *args, **kwargs):
        """Ignore."""
        pass

    @bbox.setter
    def bbox(self, *args, **kwargs):
        """Ignore."""
        pass

    @datetime.setter
    def datetime(self, *args, **kwargs):
        """Ignore."""
        pass
