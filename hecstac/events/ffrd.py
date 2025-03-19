"""Class for event items."""

import json
import logging
import os
from datetime import datetime
from typing import List

import numpy as np
from pystac import Item, Link
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.storage import StorageExtension
from shapely import to_geojson, union_all
from shapely.geometry import shape

from hecstac.common.asset_factory import AssetFactory
from hecstac.hms.assets import HMS_EXTENSION_MAPPING
from hecstac.ras.assets import RAS_EXTENSION_MAPPING

logger = logging.getLogger(__name__)


class FFRDEventItem(Item):
    """Class for event items."""

    FFRD_REALIZATION = "FFRD:realization"
    FFRD_BLOCK_GROUP = "FFRD:block_group"
    FFRD_EVENT = "FFRD:event"

    def __init__(
        self,
        realization: str,
        block_group: str,
        event_id: str,
        source_model_items: List[Item],
        hms_simulation_files: list = [],
        ras_simulation_files: list = [],
    ) -> None:
        self.realization = realization
        self.block_group = block_group
        self.event_id = event_id
        self.source_model_items = source_model_items
        self.stac_extensions = None
        self.hms_simulation_files = hms_simulation_files
        self.ras_simulation_files = ras_simulation_files
        self.hms_factory = AssetFactory(HMS_EXTENSION_MAPPING)
        self.ras_factory = AssetFactory(RAS_EXTENSION_MAPPING)
        # TODO: Add ras_factory

        super().__init__(
            self._item_id,
            self._geometry,
            self._bbox,
            self._datetime,
            self._properties,
            href=self._href,
        )

        for fpath in self.hms_simulation_files:
            self.add_hms_asset(fpath, item_type="event")

        for fpath in self.ras_simulation_files:
            self.add_ras_asset(fpath)

        self._register_extensions()
        self._add_model_links()

    def _register_extensions(self) -> None:
        ProjectionExtension.add_to(self)
        StorageExtension.add_to(self)

    def _add_model_links(self) -> None:
        """Add links to the model items."""
        for item in self.source_model_items:
            logger.info(f"Adding link from source model item: {item.id}")
            link = Link(
                rel="derived_from",
                target=item,
                title=f"Source Models: {item.id}",
            )
            self.add_link(link)

    @property
    def _item_id(self) -> str:
        """The event id for the FFRD Event STAC item."""
        return f"{self.realization}-{self.block_group}-{self.event_id}"

    @property
    def _href(self) -> str:
        return None

    @property
    def _properties(self):
        """Properties for the HMS STAC item."""
        properties = {}
        properties[self.FFRD_REALIZATION] = self.realization
        properties[self.FFRD_EVENT] = self.event_id
        properties[self.FFRD_BLOCK_GROUP] = self.block_group
        # TODO: Pull this from the items list
        # properties["proj:code"] = self.pf.basins[0].epsg
        # properties["proj:wkt"] = self.pf.basins[0].wkt
        return properties

    @property
    def _geometry(self) -> dict | None:
        """Geometry of the FFRD Event STAC item. Union of all basins in the FFRD Event items."""
        geometries = [shape(item.geometry) for item in self.source_model_items]
        return json.loads(to_geojson(union_all(geometries)))

    @property
    def _datetime(self) -> datetime:
        """The datetime for the FFRD Event STAC item."""
        return datetime.now()

    @property
    def _bbox(self) -> list[float]:
        """Bounding box of the FFRD Event STAC item."""
        bboxes = np.array([item.bbox for item in self.source_model_items])
        bboxes = [bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max()]
        return [float(i) for i in bboxes]

    def add_hms_asset(self, fpath: str, item_type: str = "event") -> None:
        """Add an asset to the FFRD Event STAC item."""
        if os.path.exists(fpath):
            logger.info(f"Adding asset: {fpath}")
            asset = self.hms_factory.create_hms_asset(fpath, item_type=item_type)
            if asset is not None:
                self.add_asset(asset.title, asset)

    def add_ras_asset(self, fpath: str) -> None:
        """Add an asset to the FFRD Event STAC item."""
        if os.path.exists(fpath):
            logger.info(f"Adding asset: {fpath}")
            asset = self.ras_factory.create_ras_asset(fpath)
            if asset is not None:
                self.add_asset(asset.title, asset)
