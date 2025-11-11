"""Class for event items."""

import json
import logging
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List

from functools import cached_property
import numpy as np
import pandas as pd
from pystac import Asset, Item, Link
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.storage import StorageExtension
from rashdf import RasPlanHdf
from shapely import simplify, to_geojson, union_all
from shapely.geometry import shape

from hecstac.common.base_io import ModelFileReader
from hecstac.common.asset_factory import AssetFactory
from hecstac.common.logger import get_logger
from hecstac.hms.assets import HMS_EXTENSION_MAPPING
from hecstac.ras.assets import RAS_EXTENSION_MAPPING
from hecstac.events.ts_utils import save_bc_lines, save_reference_lines, save_reference_points

logger = get_logger(__name__)

PARQUET_MEDIA_TYPE = "application/x-parquet"


class FFRDEventItem(Item):
    """Class for event items."""

    FFRD_REALIZATION = "FFRD:realization"
    FFRD_BLOCK_GROUP = "FFRD:block_group"
    FFRD_EVENT = "FFRD:event"

    def __init__(
        self,
        ras_simulation_files: list,
        source_model_paths: list = None,
        event_id: str = None,
        realization: str = None,
        block_group: str = None,
        hms_simulation_files: list = None,
    ) -> None:
        self.realization = realization
        self.block_group = block_group
        self.source_model_paths = source_model_paths
        self.source_model_items = []
        self.stac_extensions = None
        self.hms_simulation_files = hms_simulation_files
        self.ras_simulation_files = ras_simulation_files
        self.event_id = event_id or self._event_id_from_plan_hdf()
        self.hms_factory = AssetFactory(HMS_EXTENSION_MAPPING)
        self.ras_factory = AssetFactory(RAS_EXTENSION_MAPPING)

        if source_model_paths is not None:
            for path in source_model_paths:
                ras_model_dict = json.loads((ModelFileReader(path).content))
                self.source_model_items.append(Item.from_dict(ras_model_dict))

        super().__init__(
            self._item_id,
            self._geometry,
            self._bbox,
            self._datetime,
            self._properties,
            href=None,
        )

        for fpath in self.hms_simulation_files or []:
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

    def _event_id_from_plan_hdf(self):
        return self.plan_hdf.get_plan_info_attrs()["Plan Name"]

    @cached_property
    def plan_hdf(self):
        """Returns the main RAS plan HDF file from the ras simulation files."""
        plan_hdf_pattern = re.compile(r"p\d{2}\.hdf$")
        for ras_file in self.ras_simulation_files:
            if plan_hdf_pattern.search(ras_file):
                logger.info(f"Using {ras_file} as main plan hdf file.")
                return RasPlanHdf.open_uri(
                    ras_file, fsspec_kwargs={"default_cache_type": "blockcache", "default_block_size": 10**5}
                )

        raise ValueError("No plan HDF file found.")

    @property
    def _item_id(self) -> str:
        """The event id for the FFRD Event STAC item."""
        if self.realization and self.block_group and self.event_id:
            return f"{self.realization}-{self.block_group}-{self.event_id}"
        else:
            return self.event_id

    @property
    def _properties(self):
        """Properties for the HMS STAC item."""
        properties = {}
        properties[self.FFRD_EVENT] = self.event_id

        if self.realization:
            properties[self.FFRD_REALIZATION] = self.realization
        if self.block_group:
            properties[self.FFRD_BLOCK_GROUP] = self.block_group
        return properties

    @property
    def _geometry(self) -> dict | None:
        """Geometry of the FFRD Event STAC item. Union of all basins in the FFRD Event items."""
        if len(self.source_model_items) == 0:
            areas = self.plan_hdf.mesh_areas()

            if areas.crs is not None:
                areas = areas.to_crs("EPSG:4326")
            else:
                logger.warning("No CRS found with mesh areas, using geometry without reprojection.")

            unioned_geometry = union_all(areas.geometry)
            unioned_geometry = simplify(unioned_geometry, 0.001)
            return json.loads(to_geojson(unioned_geometry))
        else:
            geometries = [shape(item.geometry) for item in self.source_model_items]
            return json.loads(to_geojson(union_all(geometries)))

    @property
    def _datetime(self) -> datetime:
        """The datetime for the FFRD Event STAC item."""
        return datetime.now()

    @property
    def _bbox(self) -> list[float]:
        """Bounding box of the FFRD Event STAC item."""
        if len(self.source_model_items) == 0:
            areas = self.plan_hdf.mesh_areas()
            if areas.crs is not None:
                areas = areas.to_crs("EPSG:4326")
            bounds = areas.total_bounds
            return [float(b) for b in bounds]
        elif len(self.source_model_items) > 1:
            bboxes = np.array([item.bbox for item in self.source_model_items])
            bboxes = [bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max()]
            return [float(i) for i in bboxes]
        else:
            return self.source_model_items[0].bbox

    def add_hms_asset(self, fpath: str, item_type: str = "event") -> None:
        """Add an asset to the FFRD Event STAC item."""
        if os.path.exists(fpath):
            logger.info(f"Adding asset: {fpath}")
            asset = self.hms_factory.create_hms_asset(fpath, item_type=item_type)
            if asset is not None:
                self.add_asset(asset.title, asset)

    def add_ras_asset(self, fpath: str) -> None:
        """Add an asset to the FFRD Event STAC item."""
        logger.info(f"Adding asset: {fpath}")
        asset = Asset(href=fpath, title=Path(fpath).name)
        asset = self.ras_factory.asset_from_dict(asset)
        if asset is not None:
            self.add_asset(asset.title, asset)

    def _add_ts_assets_from_dict(self, asset_dict: dict, description: str):
        """Add time data as item assets."""
        for name, paths in asset_dict.items():
            for path in paths:
                var_type = path.split("/")[-1].split(".")[0]
                title = f"{name}-{var_type}"
                self.add_asset(
                    title,
                    Asset(
                        href=path,
                        title=title,
                        description=description,
                        media_type=PARQUET_MEDIA_TYPE,
                        roles=["data"],
                    ),
                )

    def add_ts_assets(self, prefix: str):
        """
        Extract and add time series assets from the plan HDF file to the STAC item. Seperates data by element and data type (stage and flow) and saves as Parquet files for each individual element and data type.

        This includes:
        - Boundary condition line time series
        - Reference line time series
        - Reference point time series

        Args:
            prefix (str): S3 or local path prefix where the time series assets will be stored.
        """
        plan_hdf = self.plan_hdf

        # Boundary condition lines
        logger.info("Processing BC Line time series data.")
        bc_ln_paths = save_bc_lines(plan_hdf, prefix)
        if bc_ln_paths:
            self._add_ts_assets_from_dict(bc_ln_paths, "Parquet containing bc line time series data.")
        else:
            logger.info("No bc lines found.")

        # Reference lines
        logger.info("Processing reference line time series data.")
        refln_paths = save_reference_lines(plan_hdf, prefix)
        if refln_paths:
            self._add_ts_assets_from_dict(refln_paths, "Parquet containing reference line time series data.")
        else:
            logger.info("No reference lines found.")

        # Reference points
        logger.info("Processing reference point time series data.")
        refpt_paths = save_reference_points(plan_hdf, prefix)
        if refpt_paths:
            self._add_ts_assets_from_dict(refpt_paths, "Parquet containing reference point time series data.")
        else:
            logger.info("No reference points found.")

    def add_flow_ts_asset(self, output_path: str):
        """Extract flow time series from boundary condition lines and reference lines. All data is combined into a single Parquet file and saved as an asset."""
        plan_hdf = self.plan_hdf

        bcline = plan_hdf.bc_lines_flow(use_names=True)
        ref_ln_flow = plan_hdf.reference_lines_flow(use_names=True)

        all_flows = pd.concat([bcline, ref_ln_flow], axis=1)

        # BC and reference lines can sometimes have same point, remove duplicates
        all_flows = all_flows.loc[:, ~all_flows.columns.duplicated(keep="first")]

        all_flows.to_parquet(output_path)

        self.add_asset(
            "flow_data",
            Asset(
                href=output_path,
                title="flow_data",
                description="Parquet containing time series flow data for reference lines and bc lines.",
                media_type=PARQUET_MEDIA_TYPE,
                roles=["data"],
            ),
        )

    def add_stage_ts_asset(self, output_path: str):
        """Extract stage time series from boundary condition lines, reference lines, and reference points. All data is combined into a single Parquet file and saved as an asset."""
        plan_hdf = self.plan_hdf

        bc_line_ts = plan_hdf.bc_lines_timeseries_output()
        ref_line_ts = plan_hdf.reference_lines_timeseries_output()

        bc_line_stage = (
            bc_line_ts["Stage"].to_dataframe().reset_index().pivot(index="time", columns="bc_line_name", values="Stage")
        )
        ref_line_stage = (
            ref_line_ts["Water Surface"]
            .to_dataframe()
            .reset_index()
            .pivot(index="time", columns="refln_name", values="Water Surface")
        )

        ref_pnt_stage = plan_hdf.reference_points_stage(use_names=True)

        all_stages = pd.concat([bc_line_stage, ref_pnt_stage, ref_line_stage], axis=1)

        # BC and reference lines can sometimes have same point, remove duplicates
        all_stages = all_stages.loc[:, ~all_stages.columns.duplicated(keep="first")]

        all_stages.to_parquet(output_path)

        self.add_asset(
            "stage_data",
            Asset(
                href=output_path,
                title="stage_data",
                description="Parquet containing time series stage data for reference lines, reference points, and bc lines.",
                media_type=PARQUET_MEDIA_TYPE,
                roles=["data"],
            ),
        )
