"""HEC-RAS STAC Item class."""

import datetime
import json
from functools import cached_property, lru_cache
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
from hecstac.common.logger import get_logger
from hecstac.common.path_manager import LocalPathManager
from hecstac.ras.assets import RAS_EXTENSION_MAPPING, GeometryAsset, GeometryHdfAsset
from hecstac.ras.consts import NULL_DATETIME, NULL_STAC_BBOX, NULL_STAC_GEOMETRY
from hecstac.ras.parser import ProjectFile
from hecstac.ras.utils import find_model_files


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
        self.logger = get_logger(__name__)

    @classmethod
    def from_prj(
        cls, ras_project_file, item_id: str, crs: str = None, simplify_geometry: bool = True, assets: list = None
    ):
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

        # href = pm.item_path(item_id)
        if not assets:
            href = pm.item_path(item_id)
            assets = {Path(i).name: Asset(i, Path(i).name) for i in find_model_files(ras_project_file)}
        else:
            href = ras_project_file.replace(".prj", ".json")
            assets = {Path(i).name: Asset(i, Path(i).name) for i in assets}
        stac = cls(
            Path(ras_project_file).stem,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            {"project_file_name": Path(ras_project_file).name},
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

    @cached_property
    def has_2d(self) -> bool:
        """Whether any geometry file has 2D elements."""
        return any([a.has_2d for a in self.geometry_assets])

    @cached_property
    def has_1d(self) -> bool:
        """Whether any geometry file has 2D elements."""
        return any([a.has_1d for a in self.geometry_assets])

    @cached_property
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
        if hasattr(self, "_geometry_cached"):
            return self._geometry_cached

        if self.crs is None:
            self.logger.warning("Geometry requested for model with no spatial reference.")
            self._geometry_cached = NULL_STAC_GEOMETRY
            return self._geometry_cached

        hdf_geom_assets = [asset for asset in self.geometry_assets if isinstance(asset, GeometryHdfAsset)]
        if len(hdf_geom_assets) == 0:
            self.logger.error("No geometry found for RAS item.")
            self._geometry_cached = NULL_STAC_GEOMETRY
            return self._geometry_cached

        geometries = []
        for i in hdf_geom_assets:
            self.logger.debug(f"Processing geometry from {i.href}")
            try:
                geometries.append(i.geometry_wgs84)
            except Exception as e:
                self.logger.error(e)
                continue

        unioned_geometry = union_all(geometries)
        if self.simplify_geometry:
            unioned_geometry = simplify(unioned_geometry, 0.001)
            if isinstance(unioned_geometry, Polygon):
                if unioned_geometry.interiors:
                    unioned_geometry = Polygon(list(unioned_geometry.exterior.coords))

        self._geometry_cached = json.loads(to_geojson(unioned_geometry))
        return self._geometry_cached

    @property
    def bbox(self) -> list[float]:
        """Get the bounding box of the model geometry."""
        return shape(self.geometry).bounds

    def to_dict(self, *args, lightweight=True, **kwargs):
        """Preload fields before serializing to dict.

        If lightweight=True, skip loading heavy geometry assets.
        """
        if not lightweight:
            _ = self.geometry
            _ = self.bbox
        _ = self.datetime
        _ = self.properties
        return super().to_dict(*args, **kwargs)

    @property
    def properties(self) -> dict:
        """Properties for the RAS STAC item."""
        if hasattr(self, "_properties_cached"):
            return self._properties_cached

        if self.ras_project_file is None:
            self._properties_cached = self._properties
            return self._properties_cached

        properties = dict(self._properties)  # Make a copy to avoid side effects

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

        self._properties_cached = properties
        return self._properties_cached

    @properties.setter
    def properties(self, value):
        """Manually setting properties clears the cache."""
        self._properties = value
        if hasattr(self, "_properties_cached"):
            del self._properties_cached

    @property
    def datetime(self) -> datetime.datetime | None:
        """Parse datetime from model geometry and return result."""
        if hasattr(self, "_datetime_cached"):
            return self._datetime_cached

        datetimes = []
        for i in self.geometry_assets:
            dt = i.file.geometry_time
            if dt is None:
                continue
            if isinstance(dt, list):
                datetimes.extend([t for t in dt if t])
            elif isinstance(dt, datetime.datetime):
                datetimes.append(dt)

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
            self.logger.warning(f"Could not extract item datetime from geometry.")
            item_time = datetime.datetime.now()
            self._properties[self.RAS_DATETIME_SOURCE] = "processing_time"

        self._datetime_cached = item_time
        return item_time

    @datetime.setter
    def datetime(self, value):
        """Ignore external setting of datetime."""
        pass

    def add_model_thumbnails(
        self, layers: list, title_prefix: str = "Model_Thumbnail", thumbnail_dir=None, s3_thumbnail_dst=None
    ):
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
        elif s3_thumbnail_dst:
            thumbnail_dest = s3_thumbnail_dst

        else:
            self.logger.warning(f"No thumbnail directory provided.  Using item directory {self.self_href}")
            thumbnail_dest = self.self_href

        for geom in self.geometry_assets:
            if isinstance(geom, GeometryHdfAsset) and geom.has_2d:
                self.logger.info(f"Writing: {thumbnail_dest}")
                self.assets[f"{geom.href.rsplit('/')[-1]}_thumbnail"] = geom.thumbnail(
                    layers=layers, title=title_prefix, thumbnail_dest=thumbnail_dest
                )

        # TODO: Add 1d model thumbnails

    def add_asset(self, key, asset):
        """Subclass asset then add, eagerly load metadata safely."""
        subclass = self.factory.asset_from_dict(asset)
        if subclass is None:
            return

        # Eager load extra fields
        _ = subclass.extra_fields

        # Safely load file only if __file_class__ is not None
        if getattr(subclass, "__file_class__", None) is not None:
            _ = subclass.file

        if self.crs is None and isinstance(subclass, GeometryHdfAsset) and subclass.file.projection is not None:
            self.crs = subclass.file.projection

        return super().add_asset(key, subclass)

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
