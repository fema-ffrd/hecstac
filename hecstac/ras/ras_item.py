import datetime
import json
import os
from enum import Enum
from functools import lru_cache
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


class ThumbnailParameter(Enum):
    XS = "cross_sections"
    MESH_2D = "mesh_areas"
    EXAMPLE = "example"


class RasModelItem(Item):
    def __init__(self, href: str, prj_file: str):
        id = ""
        geometry = NULL_STAC_GEOMETRY
        bbox = NULL_STAC_BBOX
        datetime = NULL_DATETIME
        start_datetime = None
        end_datetime = None
        properties = {}
        # add ras stac extension link
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
        self._project: ProjectAsset | None = None
        self._plan_files = []
        self._geom_files = []
        self._flow_files = []

    @property
    @lru_cache
    def geometry(self) -> dict[str, Any]:
        """
        gets geometry using either list of geom assets or list of hdf assets, perhaps simplified to a given tolerance to reduce replication of data
        (ie item would record simplified geometry used when searching collection,
        asset would have more exact geometry representing contents of geom or hdf files)
        """
        # if geometry is equal to null placeholder, continue, else return current value
        if self.geometry == NULL_STAC_GEOMETRY:
            # if hdf file is present, get mesh areas from model, simplify, and use as geometry
            # if hdf file is not present, get concave hull of cross sections and use as geometry
            pass

    def autofind_project_assets(self) -> None:
        # searches directory of prj file for files to parse as assets associated with project, then adds these as assets, storing the project asset as self.project when found
        parent = os.path.dirname(self.prj_file)
        for entry in os.scandir(parent):
            self.add_asset(entry.path)

    def add_asset(self, url: str) -> None:
        """Add an asset to the item."""
        asset = asset_factory(url)
        super().add_asset(asset.title, asset)
        if isinstance(asset, ProjectAsset):
            if self._project is not None:
                f"Only one project asset is allowed. Found {str(asset)} when {str(self._project)} was already set."
            self._project = asset
        elif isinstance(asset, (PlanAsset, PlanHdfAsset)):
            self._plan_files.append(asset)
        elif isinstance(asset, (GeometryAsset, GeomHdfAsset)):
            self._geom_files.append(asset)
        elif isinstance(asset, (SteadyFlowAsset, QuasiUnsteadyFlowAsset, UnsteadyFlowAsset)):
            self._flow_files.append(asset)

    @property
    def project_file(self) -> ProjectAsset:
        # checks if project has been set and if not, autofinds project assets
        if self._project == None:
            self.autofind_project_assets()
        # if project is still None after autofinding, raise exception
        if self._project == None:
            raise FileNotFoundError(f"no project file found")
        # after autofind has been run (building all assets associated with the item in the progress), create links between project asset and assets it references
        self._project.create_links()
        return self._project

    @property
    def plan_files(self) -> list[PlanAsset | PlanHdfAsset]:
        pass

    @property
    def geometry_files(self) -> list[GeometryAsset | GeomHdfAsset]:
        pass

    @property
    def flow_files(self) -> list[SteadyFlowAsset | QuasiUnsteadyFlowAsset | UnsteadyFlowAsset]:
        pass

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
