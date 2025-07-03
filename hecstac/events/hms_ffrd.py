import json
from pystac import Asset, Item, Link
from hecstac.events.constants import PROJ
from hecstac.common.base_io import ModelFileReader
from shapely import to_geojson, union_all
from shapely.geometry import shape
import numpy as np
from datetime import datetime
import re


class HMSEventItem(Item):
    """
    STAC Item subclass for representing an HMS event.

    This class builds a STAC Item from simulation metadata, linking to assets such as
    control files, basin files, and simulation outputs based on FEMA FFRD S3 file structure.
    """

    BASIN_PREFIX = "s3://ffrd-trinity/conformance/hydrology/trinity/"
    CONTROL_PREFIX = "s3://ffrd-trinity/conformance/hydrology/trinity/data/controlspecs/"
    SST_DSS_TEMPLATE = "s3://trinity-pilot/conformance/simulations/event-data/{}/hydrology/SST.dss"
    HDF_TEMPLATE = (
        "s3://trinity-pilot/conformance/simulations/event-data/{}/hydrology/exported-precip_trinity.p01.tmp.hdf"
    )
    SIM_DATA_TEMPLATE = "s3://trinity-pilot/cloud-hms-db/simulations/**/storm_id={}/event_id={}/*.pq"
    STORM_LINK_PREFIX = "https://stac-api.arc-apps.net/collections/72hr-events/items/"

    def __init__(
        self,
        realization: int,
        block_index: int,
        event_number: int,
        storm_fn: str,
        basin_path: str,
        storm_date: str,
        storm_id: str,
        event_index_by_block: int,
        source_model_paths: list[str],
    ) -> None:
        self.realization = realization
        self.block_index = block_index
        self.event_number = event_number
        self.storm_fn = storm_fn
        self.basin_path = basin_path
        self.storm_date = storm_date
        self.storm_id = storm_id
        self.event_index_by_block = event_index_by_block
        self.source_model_paths = source_model_paths
        self.source_model_items = []

        for path in source_model_paths:
            ras_model_dict = json.loads(ModelFileReader(path).content)
            self.source_model_items.append(Item.from_dict(ras_model_dict))

        super().__init__(
            self._item_id,
            self._geometry,
            self._bbox,
            self._datetime,
            self._properties,
            href=self._href,
        )

    def build_assets(self):
        self.add_sst_dss_asset()
        self.add_hdf_file_asset()
        self.add_basin_file_asset()
        self.add_control_file_asset()
        self.add_sim_data_asset()

    @property
    def _item_id(self) -> str:
        if self.realization and self.block_index and self.event_index_by_block:
            return f"r{self.realization:02d}-b{self.block_index:04d}-e{self.event_index_by_block:02d}"
        return str(self.event_number)

    @property
    def _geometry(self) -> dict | None:
        geometries = [shape(item.geometry) for item in self.source_model_items]
        return json.loads(to_geojson(union_all(geometries)))

    @property
    def _bbox(self) -> list[float]:
        if len(self.source_model_items) > 1:
            bboxes = np.array([item.bbox for item in self.source_model_items])
            return [
                float(bboxes[:, 0].min()),
                float(bboxes[:, 1].min()),
                float(bboxes[:, 2].max()),
                float(bboxes[:, 3].max()),
            ]
        return self.source_model_items[0].bbox

    @property
    def _datetime(self) -> datetime:
        return datetime.now()

    @property
    def _properties(self):
        return {
            "block_group": self.block_index,
            "event_id": self.event_number,
            "realization": self.realization,
            "proj:wkt2": PROJ,
            "data_time_source": "Item creation time",
        }

    @property
    def _href(self) -> str:
        return None

    def add_sst_dss_asset(self):
        full_dss_path = self.SST_DSS_TEMPLATE.format(self.event_number)
        self.add_asset(
            "complete_sim_output",
            Asset(href=full_dss_path, title="complete_sim_output", media_type="application/x-dss"),
        )

    def add_hdf_file_asset(self):
        full_hdf_path = self.HDF_TEMPLATE.format(self.event_number)
        self.add_asset(
            "excess_precip",
            Asset(href=full_hdf_path, title="excess_precip", media_type="application/x-hdf"),
        )

    def add_basin_file_asset(self):
        path = self.BASIN_PREFIX + self.basin_path
        if not path.endswith(".basin"):
            path += ".basin"
        self.add_asset("basin", Asset(href=path, title="basin", media_type="text/plain"))

    def add_control_file_asset(self):
        dt = datetime.strptime(self.storm_date, "%Y%m%d").strftime("%Y-%m-%d")
        full_path = self.CONTROL_PREFIX + dt + ".control"
        self.add_asset("control", Asset(href=full_path, title="control", media_type="text/plain"))

    def add_sim_data_asset(self):
        path = self.SIM_DATA_TEMPLATE.format(self.storm_id, self.event_number)
        self.add_asset(
            "select_sim_output",
            Asset(
                href=path,
                title="select_sim_output",
                description="Select time series extracted from hms simulation output.",
                media_type="application/x-parquet",
                extra_fields={"flow_time_series": "FLOW.pq", "base_flow_time_series": "FLOW-BASE.pq"},
            ),
        )

    def add_authoritative_model_link(self, href: str):
        self.add_link(Link(rel="derived_from", target=href, title="Source Model"))

    def add_storm_item_link(self):
        match = re.search(r"r(\d{3})\.dss", self.storm_fn)
        if not match:
            raise ValueError("Rank could not be extracted from storm file name.")
        href = self.STORM_LINK_PREFIX + match.group(1)
        self.add_link(Link(rel="derived_from", target=href, title="Storm Item"))
