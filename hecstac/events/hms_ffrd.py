"""HMS FFRD Event Item Module."""

import json
from pystac import Asset, Item, Link
from hecstac.common.base_io import ModelFileReader
from shapely import to_geojson, union_all
from shapely.geometry import shape
import numpy as np
from datetime import datetime


class HMSEventItem(Item):
    """
    STAC Item subclass for representing an FFRD HMS event.

    This class builds a STAC Item from simulation metadata, with methods to create assets such as
    control files, basin files, and simulation outputs based on FEMA FFRD S3 file structure.
    """

    def __init__(
        self,
        realization: int,
        block_index: int,
        event_number: int,
        event_index_by_block: int,
        source_model_paths: list[str],
        proj_str: str,
    ) -> None:
        self.realization = realization
        self.block_index = block_index
        self.event_number = event_number
        self.event_index_by_block = event_index_by_block
        self.source_model_paths = source_model_paths
        self.source_model_items = []
        self.proj_str = proj_str
        for path in source_model_paths:
            ras_model_dict = json.loads(ModelFileReader(path).content)
            self.source_model_items.append(Item.from_dict(ras_model_dict))
        super().__init__(
            self._item_id,
            self._geometry,
            self._bbox,
            self._datetime,
            self._properties,
            href=None,
        )

    def build_ffrd_assets(
        self,
        basin_path: str = None,
        control_file_path: str = None,
        sst_dss_path: str = None,
        hdf_path: str = None,
        sim_data_path: str = None,
    ):
        """Build the assets for the STAC item.

        Parameters
        ----------
        basin_path : str, optional
            Path to the basin file, by default None
        control_file_path : str, optional
            Path to the control file, by default None
        sst_dss_path : str, optional
            Path to the complete sim SST DSS file, by default None
        hdf_path : str, optional
            Path to the excess precip HDF file, by default None
        sim_data_path : str, optional
            Path to the simulation data file, by default None
        """
        if sst_dss_path:
            self.add_sst_dss_asset(sst_dss_path)
        if hdf_path:
            self.add_hdf_file_asset(hdf_path)
        if basin_path:
            self.add_basin_file_asset(basin_path)
        if control_file_path:
            self.add_control_file_asset(control_file_path)
        if sim_data_path:
            self.add_sim_data_asset(sim_data_path)

    @property
    def _item_id(self) -> str:
        """Generate a unique item ID based on realization, block index, and event index."""
        if self.realization and self.block_index and self.event_index_by_block:
            return f"r{self.realization:02d}-b{self.block_index:04d}-e{self.event_index_by_block:02d}"
        return str(self.event_number)

    @property
    def _geometry(self) -> dict | None:
        """Calculate the geometry for the item based on source model items."""
        geometries = [shape(item.geometry) for item in self.source_model_items]
        return json.loads(to_geojson(union_all(geometries)))

    @property
    def _bbox(self) -> list[float]:
        """Calculate the bounding box for the item based on source model items."""
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
        """Use item creation time as the datetime."""
        return datetime.now()

    @property
    def _properties(self):
        """Build the properties for the STAC item."""
        return {
            "block_group": self.block_index,
            "event_id": self.event_number,
            "realization": self.realization,
            "proj:wkt2": self.proj_str,
            "data_time_source": "Item creation time",
        }

    def add_sst_dss_asset(self, full_dss_path: str):
        """Add complete sim SST DSS file as an asset."""
        self.add_asset(
            "complete_sim_output",
            Asset(href=full_dss_path, title="complete_sim_output", media_type="application/x-dss"),
        )

    def add_hdf_file_asset(self, full_hdf_path: str):
        """Add excess precip HDF file as an asset."""
        self.add_asset(
            "excess_precip",
            Asset(href=full_hdf_path, title="excess_precip", media_type="application/x-hdf"),
        )

    def add_basin_file_asset(self, full_basin_path: str):
        """Add basin file as an asset."""
        if not full_basin_path.endswith(".basin"):
            full_basin_path += ".basin"
        self.add_asset("basin", Asset(href=full_basin_path, title="basin", media_type="text/plain"))

    def add_control_file_asset(self, control_file_path: str):
        """Add control file as an asset."""
        self.add_asset("control", Asset(href=control_file_path, title="control", media_type="text/plain"))

    def add_sim_data_asset(self, sim_data_path: str):
        """Add simulation output data as an asset."""
        self.add_asset(
            "select_sim_output",
            Asset(
                href=sim_data_path,
                title="select_sim_output",
                description="Select time series extracted from hms simulation output.",
                media_type="application/x-parquet",
                extra_fields={"flow_time_series": "FLOW.pq", "base_flow_time_series": "FLOW-BASE.pq"},
            ),
        )

    def add_authoritative_model_link(self, item_href: str = None):
        """Add a link to the authoritative model(s). If item_href is provided, it links to that item; otherwise, it links to the source model paths."""
        if item_href:
            self.add_link(Link(rel="derived_from", target=item_href, title="Source Model"))
        else:
            for href in self.source_model_paths:
                self.add_link(Link(rel="derived_from", target=href, title="Source Model"))

    def add_storm_item_link(self, storm_item_href: str):
        """Add a link to the storm item."""
        self.add_link(Link(rel="derived_from", target=storm_item_href, title="Storm Item"))
