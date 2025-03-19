"""HEC-RAS STAC Item class."""

import datetime
import json
import logging
from functools import lru_cache
from pathlib import Path

import pystac
import pystac.errors
from pyproj import CRS
from pystac import Asset, Item
from pystac.extensions.projection import ProjectionExtension
from pystac.utils import datetime_to_str
from shapely import Polygon, simplify, to_geojson, union_all
from shapely.geometry import shape

from hecstac.common.asset_factory import AssetFactory
from hecstac.common.path_manager import LocalPathManager
from hecstac.ras.assets import RAS_EXTENSION_MAPPING, GeometryAsset, GeometryHdfAsset
from hecstac.ras.consts import NULL_DATETIME, NULL_STAC_BBOX, NULL_STAC_GEOMETRY
from hecstac.ras.parser import ProjectFile
from hecstac.ras.utils import find_model_files

# logger = logging.getLogger(__name__)


class RASModelItem(Item):
    """An object representation of a HEC-RAS model."""

    PROJECT = "HEC-RAS:project"
    PROJECT_TITLE = "HEC-RAS:project_title"
    MODEL_UNITS = "HEC-RAS:unit system"
    MODEL_GAGES = "HEC-RAS:gages"
    PROJECT_VERSION = "HEC-RAS:version"
    PROJECT_DESCRIPTION = "HEC-RAS:description"
    PROJECT_STATUS = "HEC-RAS:status"
    PROJECT_UNITS = "HEC-RAS:unit_system"

    RAS_HAS_1D = "HEC-RAS:has_1d"
    RAS_HAS_2D = "HEC-RAS:has_2d"
    RAS_DATETIME_SOURCE = "HEC-RAS:datetime_source"

    def __init__(self, *args, **kwargs):
        """Add a few default properties to the base class."""
        super().__init__(*args, **kwargs)
        self.simplify_geometry = True

    @classmethod
    def from_prj(cls, ras_project_file, item_id: str, crs: str = None, simplify_geometry: bool = True):
        """
        Create a STAC item from a HEC-RAS .prj file.

        Parameters
        ----------
        ras_project_file : str
            Path to the HEC-RAS project file (.prj).
        item_id : str
            Unique item id for the STAC item.
        crs : str, optional
            Coordinate reference system (CRS) to apply to the item. If None, the CRS will be extracted from the geometry .hdf file.
        simplify_geometry : bool, optional
            Whether to simplify geometry. Defaults to True.

        Returns
        -------
        stac : RASModelItem
            An instance of the class representing the STAC item.
        """
        pm = LocalPathManager(Path(ras_project_file).parent)

        href = pm.item_path(item_id)
        # TODO: Add option to recursively iterate through all subdirectories in a model folder.
        assets = {Path(i).name: Asset(i, Path(i).name) for i in find_model_files(ras_project_file)}

        stac = cls(
            Path(ras_project_file).stem,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            {"ras_project_file": ras_project_file},
            href=href,
            assets=assets,
        )
        if crs:
            stac.crs = crs
        stac.simplify_geometry = simplify_geometry
        stac.pm = pm

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
        return any([a.has_2d for a in self.geometry_assets])

    @property
    def has_1d(self) -> bool:
        """Whether any geometry file has 2D elements."""
        return any([a.has_1d for a in self.geometry_assets])

    @property
    def geometry_assets(self) -> list[GeometryHdfAsset | GeometryAsset]:
        """Return any RasGeomHdf in assets."""
        return [a for a in self.assets.values() if isinstance(a, (GeometryHdfAsset, GeometryAsset))]

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
        if len(self.geometry_assets) == 0:
            logging.error("No geometry found for RAS item.")
            return NULL_STAC_GEOMETRY

        geometries = []
        for i in self.geometry_assets:
            logging.debug(f"Processing geometry from {i.href}")
            try:
                geometries.append(i.geometry_wgs84)
            except Exception as e:
                logging.error(e)
                # logger.warning(f"Could not process geometry from {i.href}")
                continue

        unioned_geometry = union_all(geometries)
        if self.simplify_geometry:
            unioned_geometry = simplify(unioned_geometry, 0.001)
            if unioned_geometry.interiors:
                unioned_geometry = Polygon(list(unioned_geometry.exterior.coords))
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
        properties[self.RAS_HAS_1D] = self.has_1d
        properties[self.RAS_HAS_2D] = self.has_2d
        properties[self.PROJECT_TITLE] = self.pf.project_title
        properties[self.PROJECT_VERSION] = self.pf.ras_version
        properties[self.PROJECT_DESCRIPTION] = self.pf.project_description
        properties[self.PROJECT_STATUS] = self.pf.project_status
        properties[self.MODEL_UNITS] = self.pf.project_units
        if self.datetime is not None:
            properties["datetime"] = datetime_to_str(self.datetime)
        else:
            properties["datetime"] = None
        # TODO: once all assets are created, populate associations between assets
        return properties

    @properties.setter
    def properties(self, properties: dict):
        """Set properties."""
        self._properties = properties

    @property
    def datetime(self) -> datetime.datetime | None:
        """Parse datetime from model geometry and return result."""
        datetimes = []
        for i in self.geometry_assets:
            i = i.file.geometry_time
            if i is None:
                continue
            elif isinstance(i, list):
                datetimes.extend([j for j in i if j is not None])
            elif isinstance(i, datetime.datetime):
                datetimes.append(i)

        datetimes = list(set(datetimes))
        if len(datetimes) > 1:
            self._properties["start_datetime"] = datetime_to_str(min(datetimes))
            self._properties["end_datetime"] = datetime_to_str(max(datetimes))
            self._properties[self.RAS_DATETIME_SOURCE] = "model_geometry"
            item_time = None
        elif len(datetimes) == 1:
            item_time = datetimes[0]
            self._properties[self.RAS_DATETIME_SOURCE] = "model_geometry"
        else:
            logging.warning("Could not extract item datetime from geometry, using item processing time.")
            item_time = datetime.datetime.now()
            self._properties[self.RAS_DATETIME_SOURCE] = "processing_time"
        return item_time

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
            thumbnail_dest = self.self_href

        for geom in self.geometry_assets:
            if isinstance(geom, GeometryHdfAsset) and geom.has_2d:
                self.assets[f"{geom.href.rsplit('/')[-1]}_thumbnail"] = geom.thumbnail(
                    layers=layers, title=title_prefix, thumbnail_dest=thumbnail_dest
                )

        # TODO: Add 1d model thumbnails

    def add_asset(self, key, asset):
        """Subclass asset then add."""
        subclass = self.factory.asset_from_dict(asset)
        if subclass is None:
            return
        if self.crs is None and isinstance(subclass, GeometryHdfAsset) and subclass.file.projection is not None:
            self.crs = subclass.file.projection
        return super().add_asset(key, subclass)

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
