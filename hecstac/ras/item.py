"""HEC-RAS STAC Item class."""

import datetime
import json
import os
from functools import cached_property
from pathlib import Path

import pystac
import pystac.errors
from pyproj import CRS
from pystac import Asset, Item
from pystac.extensions.projection import ProjectionExtension
from pystac.utils import datetime_to_str
from shapely import Polygon, simplify, to_geojson, union_all
from shapely.geometry import shape

import hecstac
from hecstac.common.asset_factory import AssetFactory
from hecstac.common.base_io import ModelFileReaderError
from hecstac.common.logger import get_logger
from hecstac.common.path_manager import LocalPathManager
from hecstac.ras.assets import (
    RAS_EXTENSION_MAPPING,
    GeometryAsset,
    GeometryHdfAsset,
    PlanAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
)
from hecstac.ras.consts import NULL_DATETIME, NULL_STAC_BBOX, NULL_STAC_GEOMETRY
from hecstac.ras.parser import ProjectFile
from hecstac.ras.utils import find_model_files

logger = get_logger(__name__)


class RASModelItem(Item):
    """An object representation of a HEC-RAS model."""

    PROJECT = "HEC-RAS:project"
    PROJECT_TITLE = "HEC-RAS:project_title"
    MODEL_UNITS = "HEC-RAS:unit system"
    MODEL_GAGES = "HEC-RAS:gages"  # TODO: Is this deprecated?
    PROJECT_VERSION = "HEC-RAS:version"
    PROJECT_DESCRIPTION = "HEC-RAS:description"
    PROJECT_STATUS = "HEC-RAS:status"
    RAS_HAS_1D = "HEC-RAS:has_1d"
    RAS_HAS_2D = "HEC-RAS:has_2d"
    RAS_DATETIME_SOURCE = "HEC-RAS:datetime_source"
    HECSTAC_VERSION = "HEC-RAS:hecstac_version"

    def __init__(self, *args, **kwargs):
        """Add a few default properties to the base class."""
        super().__init__(*args, **kwargs)
        self.simplify_geometry = True

    @classmethod
    def from_prj(cls, ras_project_file: str, crs: str = None, simplify_geometry: bool = True, assets: list = None):
        """
        Create a STAC item from a HEC-RAS .prj file.

        Parameters
        ----------
        ras_project_file : str
            Path to the HEC-RAS project file (.prj).
        crs : str, optional
            Coordinate reference system (CRS) to apply to the item. If None, the CRS will be extracted from the geometry .hdf file.
        simplify_geometry : bool, optional
            Whether to simplify geometry. Defaults to True.

        Returns
        -------
        stac : RASModelItem
            An instance of the class representing the STAC item.
        """
        if not assets:
            assets = {Path(i).name: Asset(i, Path(i).name) for i in find_model_files(ras_project_file)}
        else:
            assets = {Path(i).name: Asset(i, Path(i).name) for i in assets}
        stac = cls(
            Path(ras_project_file).stem,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            {cls.PROJECT: Path(ras_project_file).name},
            href=ras_project_file.replace(".prj", ".json"),
            assets=assets,
        )
        if crs:
            stac.crs = crs
        stac.simplify_geometry = simplify_geometry
        stac.update_properties()

        return stac

    @property
    def ras_project_file(self) -> str:
        """Get the path to the HEC-RAS .prj file."""
        return self._properties.get("project_file_name")

    @property
    def factory(self) -> AssetFactory:
        """Return AssetFactory for this item."""
        return AssetFactory(RAS_EXTENSION_MAPPING)

    @property
    def pm(self) -> LocalPathManager:
        """Get the path manager rooted at project file's href."""
        return LocalPathManager(str(Path(self.project_asset.href).parent))

    @cached_property
    def project_asset(self) -> ProjectAsset:
        """Find the project file for this model."""
        return [i for i in self.assets.values() if isinstance(i, ProjectAsset)][0]

    @cached_property
    def pf(self) -> ProjectFile:
        """Get a ProjectFile instance for the RAS Model .prj file."""
        return self.project_asset.file

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

    @cached_property
    def plan_assets(self) -> list[PlanAsset]:
        """Return any RasGeomHdf in assets."""
        return [a for a in self.assets.values() if isinstance(a, PlanAsset)]

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
            logger.warning("Geometry requested for model with no spatial reference.")
            self._geometry_cached = NULL_STAC_GEOMETRY
            return self._geometry_cached

        if len(self.geometry_assets) == 0:
            logger.error("No geometry found for RAS item.")
            self._geometry_cached = NULL_STAC_GEOMETRY
            return self._geometry_cached

        geometries = []
        for i in self.geometry_assets:
            logger.debug(f"Processing geometry from {i.href}")
            try:
                geometries.append(i.geometry_wgs84)
            except Exception as e:
                logger.error(e)
                continue

        unioned_geometry = union_all(geometries)
        if self.simplify_geometry:
            unioned_geometry = simplify(unioned_geometry, 0.001)
            if isinstance(unioned_geometry, Polygon):
                if unioned_geometry.interiors:
                    unioned_geometry = Polygon(list(unioned_geometry.exterior.coords))

        self._geometry_cached = json.loads(to_geojson(unioned_geometry))
        return self._geometry_cached

    @geometry.setter
    def geometry(self, val):
        """Ignore external setting of geometry."""
        pass

    @property
    def bbox(self) -> list[float]:
        """Get the bounding box of the model geometry."""
        return list(shape(self.geometry).bounds)

    @bbox.setter
    def bbox(self, val):
        """Ignore external setting of bbox."""
        pass

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

    def update_properties(self) -> dict:
        """Force recalculation of HEC-RAS properties."""
        self.properties[self.PROJECT] = self.project_asset.name
        self.properties[self.RAS_HAS_1D] = self.has_1d
        self.properties[self.RAS_HAS_2D] = self.has_2d
        self.properties[self.PROJECT_TITLE] = self.pf.project_title
        self.properties[self.PROJECT_VERSION] = self.pf.ras_version
        self.properties[self.PROJECT_DESCRIPTION] = self.pf.project_description
        self.properties[self.PROJECT_STATUS] = self.pf.project_status
        self.properties[self.MODEL_UNITS] = self.pf.project_units
        self.properties[self.HECSTAC_VERSION] = hecstac.__version__

        datetimes = self.model_datetime
        if len(datetimes) > 1:
            self.properties["start_datetime"] = datetime_to_str(min(datetimes))
            self.properties["end_datetime"] = datetime_to_str(max(datetimes))
            self.properties[self.RAS_DATETIME_SOURCE] = "model_geometry"
            self.datetime = None
        elif len(datetimes) == 1:
            self.datetime = datetimes[0]
            self.properties[self.RAS_DATETIME_SOURCE] = "model_geometry"
        else:
            logger.warning(f"Could not extract item datetime from geometry.")
            self.datetime = datetime.datetime.now()
            self.properties[self.RAS_DATETIME_SOURCE] = "processing_time"

    @property
    def model_datetime(self) -> list[datetime.datetime]:
        """Parse datetime from model geometry and return result."""
        datetimes = []
        for i in self.geometry_assets:
            dt = i.file.geometry_time
            if dt is None:
                continue
            if isinstance(dt, list):
                datetimes.extend([t for t in dt if t])
            elif isinstance(dt, datetime.datetime):
                datetimes.append(dt)

        return list(set(datetimes))

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
            logger.warning(f"No thumbnail directory provided.  Using item directory {self.self_href}")
            thumbnail_dest = os.path.dirname(self.self_href)

        for geom in self.geometry_assets:
            if isinstance(geom, GeometryHdfAsset) and geom.has_2d:
                logger.info(f"Writing: {thumbnail_dest}")
                self.assets[f"{geom.href.rsplit('/')[-1]}_thumbnail"] = geom.thumbnail(
                    layers=layers, title=title_prefix, thumbnail_dest=thumbnail_dest
                )
            elif isinstance(geom, GeometryAsset) and not (os.path.exists(geom.href + ".hdf") and geom.has_2d):
                logger.info(f"Writing: {thumbnail_dest}")
                self.assets[f"{geom.href.rsplit('/')[-1]}_thumbnail"] = geom.thumbnail(
                    layers=layers, title=title_prefix, thumbnail_dest=thumbnail_dest
                )

    def add_model_geopackages(self, local_dst=None, s3_dst=None):
        """Generate model geopackage asset for each geometry file.

        Parameters
        ----------
        local_dst : str, optional
            Directory for created geopackages. If None then geopackages will be exported to same level as the item.
        s3_dst : str, optional
            S3 prefix for created geopackages. If None then geopackages will be exported to same level as the item.
        """
        if local_dst:
            dst = local_dst
        elif s3_dst:
            dst = s3_dst
        else:
            logger.warning(f"No thumbnail directory provided.  Using item directory {self.self_href}")
            dst = os.path.dirname(self.self_href)

        for geom in self.geometry_assets:
            # TODO: Implement hdf geopackages.
            if isinstance(geom, GeometryAsset):
                logger.info(f"Writing: {dst}")
                self.assets[f"{geom.href.rsplit('/')[-1]}_geopackage"] = geom.geopackage(dst, self.gpkg_metadata)

    @property
    def _primary_plan(self) -> PlanAsset:
        """Primary plan for use in Ripple1D."""  # TODO: develop test for this logic. easily tested
        if len(self.plan_assets) == 0:
            return None
        elif len(self.plan_assets) == 1:
            return self.plan_assets[0]

        candidate_plans = [i for i in self.plan_assets if not i.file.is_encroached]

        if len(candidate_plans) > 1:
            cur_plan = [i for i in candidate_plans if i.name == self.pf.plan_current]
            if len(cur_plan) == 0:
                return cur_plan
            else:
                return candidate_plans[0]
        elif len(candidate_plans) == 0:
            return self.plan_assets[0]
        else:
            return candidate_plans[0]

    @property
    def _primary_flow(self) -> SteadyFlowAsset | UnsteadyFlowAsset | QuasiUnsteadyFlowAsset:
        """Flow asset listed in the primary plan."""
        for i in self.assets.values():
            if isinstance(i, (SteadyFlowAsset, UnsteadyFlowAsset, QuasiUnsteadyFlowAsset)):
                if i.name == self._primary_plan.file.flow_file:
                    return i
        return None

    @property
    def _primary_geometry(self) -> GeometryAsset:
        """Geometry asset listed in the primary plan."""
        for i in self.assets.values():
            if isinstance(i, GeometryAsset):
                if i.name == self._primary_plan.file.geometry_file:
                    return i
        return None

    @property
    def gpkg_metadata(self) -> dict:
        """Generate metadata for the geopackage metadata table."""
        metadata = {}
        metadata["plans_files"] = "\n".join([i.name for i in self.assets.values() if isinstance(i, PlanAsset)])
        metadata["geom_files"] = "\n".join([i.name for i in self.geometry_assets])
        metadata["steady_flow_files"] = "\n".join(
            [i.name for i in self.assets.values() if isinstance(i, SteadyFlowAsset)]
        )
        metadata["unsteady_flow_files"] = "\n".join(
            [i.name for i in self.assets.values() if isinstance(i, UnsteadyFlowAsset)]
        )
        metadata["ras_project_file"] = self.properties[self.PROJECT]
        metadata["ras_project_title"] = self.pf.project_title
        metadata["plans_titles"] = "\n".join([i.title for i in self.assets if isinstance(i, PlanAsset)])
        metadata["geom_titles"] = "\n".join([i.title for i in self.geometry_assets])
        metadata["steady_flow_titles"] = "\n".join([i.title for i in self.assets if isinstance(i, SteadyFlowAsset)])
        metadata["active_plan"] = self.pf.plan_current
        metadata["primary_plan_file"] = self._primary_plan.name
        metadata["primary_plan_title"] = self._primary_plan.file.plan_title
        metadata["primary_flow_file"] = self._primary_flow.name
        metadata["primary_flow_title"] = self._primary_flow.file.flow_title
        metadata["primary_geom_file"] = self._primary_geometry.name
        metadata["primary_geom_title"] = self._primary_geometry.file.geom_title
        metadata["ras_version"] = self.pf.ras_version
        metadata["hecstac_version"] = hecstac.__version__
        if isinstance(self._primary_flow, SteadyFlowAsset):
            metadata["profile_names"] = "\n".join(self._primary_flow.file.profile_names)
        else:
            metadata["profile_names"] = None
        metadata["units"] = self.pf.project_units
        return metadata

    def add_asset(self, key, asset):
        """Subclass asset then add, eagerly load metadata safely."""
        subclass = self.factory.asset_from_dict(asset)
        if subclass is None:
            return

        # Eager load extra fields
        try:
            _ = subclass.extra_fields
        except ModelFileReaderError as e:
            logger.error(e)
            return

        # Safely load file only if __file_class__ is not None
        if getattr(subclass, "__file_class__", None) is not None:
            _ = subclass.file

        if self.crs is None and isinstance(subclass, GeometryHdfAsset) and subclass.file.projection is not None:
            self.crs = subclass.file.projection

        return super().add_asset(key, subclass)
