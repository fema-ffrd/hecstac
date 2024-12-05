import datetime
import json
from typing import Any

from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pystac import Item
from ras_asset import (
    GenericAsset,
    GeometryAsset,
    GeomHdfAsset,
    PlanAsset,
    PlanHdfAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
)
from shapely import Polygon, to_geojson
from stac_utils import asset_factory

from ..utils.placeholders import NULL_DATETIME, NULL_STAC_BBOX, NULL_STAC_GEOMETRY


class RasModelItem(Item):
    def __init__(self, prj_file: str):
        href = prj_file
        id = ""
        geometry = NULL_STAC_GEOMETRY
        bbox = NULL_STAC_BBOX
        datetime, start_datetime, end_datetime = (NULL_DATETIME, None, None)
        properties = {}
        stac_extensions = []
        collection = None
        extra_fields = {}
        assets = {}
        super().__init__(
            id,
            geometry,
            bbox,
            datetime,
            properties,
            start_datetime,
            end_datetime,
            stac_extensions,
            href,
            collection,
            extra_fields,
            assets,
        )
        self.prj_file = prj_file
        self._geometry: Polygon | None = None
        self._bbox: Polygon | None = None

    def _get_datetimes(self) -> tuple[datetime.datetime, datetime.datetime | None, datetime.datetime | None]:
        if isinstance(self.hdf_asset, PlanHdfAsset):
            pass
        elif isinstance(self.hdf_asset, GeomHdfAsset):
            pass

    @property
    def geometry(self) -> dict[str, Any]:
        """
        gets geometry using either list of geom assets or list of hdf assets, perhaps simplified to a given tolerance to reduce replication of data
        (ie item would record simplified geometry used when searching collection,
        asset would have more exact geometry representing contents of geom or hdf files)
        """
        if self._geometry == None:
            pass
        return json.loads(to_geojson(self._geometry))

    def autofind_project_assets(self) -> None:
        # searches directory for files to parse as assets associated with project, then adds these as assets, storing the project asset as self.project when found
        pass

    def add_asset(self, url: str) -> None:
        """Add an asset to the item."""
        asset = asset_factory(url)
        super().add_asset(asset.title, asset)
        if isinstance(asset, ProjectAsset):
            if self._project is not None:
                f"Only one project asset is allowed. Found {str(asset)} when {str(self._project)} was already set."
            self._project = asset

    @property
    def project(self) -> ProjectAsset:
        # checks if project has been set and if not, autofinds project assets
        if self._project == None:
            self.autofind_project_assets()
        # if project is still None after autofinding, raise exception
        if self._project == None:
            raise FileNotFoundError(f"no project file found")
        return self._project

    def thumbnail(self, add_asset: bool, write: bool) -> Figure:
        # create thumbnail figure
        # if add_asset or write is true, save asset to filepath relative to item href and add thumbnail asset to asset list
        pass

    def get_usgs_data(self, add_properties: bool) -> GeoDataFrame:
        # retrieve USGS gages using model reference lines from HDF asset, if available, else raise exception
        # if add_properties is true, create USGS metadata JSON item for each gage and add it to array property
        pass
