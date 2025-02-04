"""HEC-RAS STAC Item creation class."""

import datetime
import json
import logging
import os
from pathlib import Path

from pyproj import CRS, Transformer
from pystac import Item
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.storage import StorageExtension
from shapely import Geometry, Polygon, simplify, to_geojson, union_all
from shapely.ops import transform

from hecstac.common.path_manager import LocalPathManager
from hecstac.ras.parser import ProjectFile
from hecstac.ras.consts import (
    NULL_DATETIME,
    NULL_STAC_GEOMETRY,
    NULL_STAC_BBOX,
)

from hecstac.common.asset_factory import AssetFactory
from hecstac.ras.assets import (
    RAS_EXTENSION_MAPPING,
    GeometryAsset,
    GeometryHdfAsset,
    ProjectAsset,
)


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

    @classmethod
    def from_prj(cls, ras_project_file, item_id: str, crs: str = None, simplify_geometry: bool = True):
        """Create an item from a RAS .prj file."""
        stac.pf = ProjectFile(ras_project_file)
        stac.has_2d = False
        stac.has_1d = False
        stac._geom_files = []
        stac.crs = crs
        stac.factory = AssetFactory(RAS_EXTENSION_MAPPING)

        properties = {"ras_project_file": ras_project_file}
        pm = LocalPathManager(Path(ras_project_file).parent)
        stac.pm = pm

        href = pm.item_path(item_id)
        stac._href = href

        stac = cls(
            Path(ras_project_file).stem,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            properties,
            href=href,
        )

        ras_asset_files = stac.scan_model_dir(ras_project_file)

        for fpath in ras_asset_files:
            if fpath and fpath != href:
                logging.info(f"Processing asset: {fpath}")
                stac.add_ras_asset(fpath)

        stac.geometry, stac.bbox = stac.add_geometry(simplify_geometry)
        stac.properties.update(stac.add_properties)
        stac.datetime = stac._datetime
        if stac.crs:
            stac.apply_projection_extension(stac.crs)

        return stac

    @property
    def add_properties(self) -> None:
        """Properties for the RAS STAC item."""
        properties = {}
        # properties[self.RAS_HAS_1D] = self.has_1d
        properties[self.RAS_HAS_2D] = self.has_2d
        properties[self.PROJECT_TITLE] = self.pf.project_title
        properties[self.PROJECT_VERSION] = self.pf.ras_version
        properties[self.PROJECT_DESCRIPTION] = self.pf.project_description
        properties[self.PROJECT_STATUS] = self.pf.project_status
        properties[self.MODEL_UNITS] = self.pf.project_units

        # TODO: once all assets are created, populate associations between assets
        return properties

    def add_geometry(self, simplify_geometry: bool) -> dict | None:
        """Parse geometries from 2d hdf files and updates the stac item geometry, simplifying them if needed."""
        geometries = []

        if self.has_2d:
            geometries.append(self.parse_2d_geom())

        # if self.has_1d:
        #     geometries.append(self.parse_1d_geom())

        if len(geometries) == 0:
            logging.error("No geometry found for RAS item.")
            return NULL_STAC_GEOMETRY, NULL_STAC_BBOX

        unioned_geometry = union_all(geometries)
        if simplify_geometry:
            unioned_geometry = simplify(unioned_geometry, 0.001)

        geometry = json.loads(to_geojson(unioned_geometry))
        bbox = unioned_geometry.bounds

        return geometry, bbox

    @property
    def _datetime(self) -> datetime:
        """The datetime for the RAS STAC item."""
        item_datetime = None

        for geom_file in self._geom_files:
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
        """Generate model thumbnail asset for each geometry file.

        Parameters
        ----------
        layers : list
            List of geometry layers to be included in the plot. Options include 'mesh_areas', 'breaklines', 'bc_lines'
        title_prefix : str, optional
            Thumbnail title prefix, by default "Model_Thumbnail".
        thumbnail_dir : str, optional
            Directory for created thumbnails. If None then thumbnails will be exported to same level as the item.
        """
        if thumbnail_dir:
            thumbnail_dest = thumbnail_dir
        else:
            thumbnail_dest = self._href

        for geom in self._geom_files:
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
            logging.debug(f"Adding asset {str(asset)}")
        except TypeError as e:
            logging.error(f"Error creating asset for {fpath}: {e}")
            return

        if asset:
            self.add_asset(asset.title, asset)

            if isinstance(asset, GeometryHdfAsset):
                # if item and asset crs are None, pass and use null geometry
                if self.crs is None and asset.crs is None:
                    pass
                # Use asset crs as item crs if there is no item crs
                elif self.crs is None and asset.crs is not None:
                    self.crs = asset.crs
                # If item has crs, use it as the asset crs
                elif self.crs:
                    asset.crs = self.crs

                if asset.check_2d:
                    self._geom_files = []
                    self._geom_files.append(asset)
                    self.has_2d = True
                    self.properties[self.RAS_HAS_2D] = True
            # elif isinstance(asset, GeometryAsset):
            # if asset.geomf.has_1d:
            #     self.has_1d = False TODO: Implement 1d functionality
            #     self.properties[self.RAS_HAS_1D] = True
            #     self._geom_files.append(asset)

    def _geometry_to_wgs84(self, geom: Geometry) -> Geometry:
        """Convert geometry CRS to EPSG:4326 for stac item geometry."""
        pyproj_crs = CRS.from_user_input(self.crs)
        wgs_crs = CRS.from_authority("EPSG", "4326")
        if pyproj_crs != wgs_crs:
            transformer = Transformer.from_crs(pyproj_crs, wgs_crs, True)
            return transform(transformer.transform, geom)
        return geom

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

    def scan_model_dir(self, ras_project_file):
        """Find all files in the project folder."""
        base_dir = os.path.dirname(ras_project_file)
        files = []
        for root, _, filenames in os.walk(base_dir):
            depth = root[len(base_dir) :].count(os.sep)
            if depth > 1:
                break
            for filename in filenames:
                files.append(os.path.join(root, filename))
        return files

    def apply_projection_extension(self, crs: str):
        """Apply the projection extension to this item given a CRS."""
        prj_ext = ProjectionExtension.ext(self, add_if_missing=True)
        og_crs = CRS(crs)
        prj_ext.apply(epsg=og_crs.to_epsg(), wkt2=og_crs.to_wkt())
