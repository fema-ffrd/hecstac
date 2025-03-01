"""HEC-RAS STAC Item class."""

import json
import logging
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import requests
from pystac import Asset, Item
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.storage import StorageExtension
from shapely import to_geojson, unary_union

from hecstac.common.asset_factory import AssetFactory
from hecstac.common.path_manager import LocalPathManager
from hecstac.hms.assets import HMS_EXTENSION_MAPPING
from hecstac.hms.parser import BasinFile, ProjectFile
from hecstac.ras.consts import NULL_DATETIME, NULL_STAC_BBOX, NULL_STAC_GEOMETRY

logger = logging.getLogger(__name__)


class HMSModelItem(Item):
    """An object representation of a HEC-HMS model."""

    PROJECT = "hms:project"
    PROJECT_TITLE = "hms:project_title"
    MODEL_UNITS = "hms:unit system"
    MODEL_GAGES = "hms:gages"
    PROJECT_VERSION = "hms:version"
    PROJECT_DESCRIPTION = "hms:description"
    PROJECT_UNITS = "hms:unit_system"
    SUMMARY = "hms:summary"

    def __init__(self, *args, **kwargs):
        """Add a few default properties to the base class."""
        super().__init__(*args, **kwargs)
        self.simplify_geometry = True

    @classmethod
    def from_prj(cls, hms_project_file, item_id: str, simplify_geometry: bool = True):
        """
        Create an `HMSModelItem` from a HEC-HMS project file.

        Parameters
        ----------
        hms_project_file : str
            Path to the HEC-HMS project file (.hms).
        item_id : str
            Unique item ID for the STAC item.
        simplify_geometry : bool, optional
            Whether to simplify geometry. Defaults to True.

        Returns
        -------
        stac : HMSModelItem
            An instance of the class representing the STAC item.
        """
        pm = LocalPathManager(Path(hms_project_file).parent)
        href = pm.item_path(item_id)
        pf = ProjectFile(hms_project_file, assert_uniform_version=False)
        # Create GeoJSON and Thumbnails
        cls._check_files_exists(cls, pf.files + pf.rasters)
        geojson_paths = cls.write_element_geojsons(cls, pf.basins, pm, pf)
        thumbnail_paths = cls.make_thumbnails(cls, pf.basins, pm)

        # Collect all assets
        assets = {Path(i).name: Asset(i) for i in pf.files + pf.rasters + geojson_paths + thumbnail_paths}
        # Create the STAC Item
        stac = cls(
            Path(hms_project_file).stem,
            NULL_STAC_GEOMETRY,
            NULL_STAC_BBOX,
            NULL_DATETIME,
            {"hms_project_file": hms_project_file},
            href=href,
            assets=assets,
        )
        stac.pm = pm
        stac.simplify_geometry = simplify_geometry

        stac._register_extensions()
        return stac

    def _register_extensions(self) -> None:
        ProjectionExtension.add_to(self)
        StorageExtension.add_to(self)

    @property
    def hms_project_file(self) -> str:
        """Get the path to the HEC-HMS .hms file."""
        return self._properties.get("hms_project_file")

    @property
    @lru_cache
    def factory(self) -> AssetFactory:
        """Return AssetFactory for this item."""
        return AssetFactory(HMS_EXTENSION_MAPPING)

    @property
    @lru_cache
    def pf(self) -> ProjectFile:
        """Get a ProjectFile instance for the HMS Model .hms file."""
        return ProjectFile(self.hms_project_file)

    @property
    def properties(self) -> dict:
        """Properties for the HMS STAC item."""
        properties = self._properties
        properties[self.PROJECT] = f"{self.pf.name}.hms"
        properties[self.PROJECT_TITLE] = self.pf.name
        properties[self.PROJECT_VERSION] = self.pf.attrs["Version"]
        properties[self.PROJECT_DESCRIPTION] = self.pf.attrs.get("Description")

        # Get data from the first basin
        properties[self.MODEL_UNITS] = self.pf.basins[0].attrs["Unit System"]
        properties[self.MODEL_GAGES] = self.pf.basins[0].gages
        properties["proj:code"] = self.pf.basins[0].epsg

        if self.pf.basins[0].epsg is None:
            logger.warning("No EPSG code found in basin file.")

        properties["proj:wkt"] = self.pf.basins[0].wkt
        properties[self.SUMMARY] = self.pf.file_counts

        return properties

    @properties.setter
    def properties(self, properties: dict):
        """Set properties."""
        self._properties = properties

    @property
    def geometry_assets(self) -> list[BasinFile]:
        """Return list of basin geometry assets."""
        return self.pf.basins

    @property
    def geometry(self) -> dict:
        """Return footprint of the model as a GeoJSON."""
        if not self.geometry_assets:
            return NULL_STAC_GEOMETRY

        geometries = [
            b.basin_geom.simplify(0.001) if self.simplify_geometry else b.basin_geom for b in self.geometry_assets
        ]
        unioned_geometry = unary_union(geometries)

        return json.loads(to_geojson(unioned_geometry))

    @property
    def bbox(self) -> list[float]:
        """Bounding box of the HMS model."""
        if not self.geometry_assets:
            return NULL_STAC_BBOX

        bboxes = np.array([b.bbox(4326) for b in self.geometry_assets])
        return [float(i) for i in [bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max()]]

    @property
    def datetime(self) -> datetime:
        """The datetime for the HMS STAC item."""
        date = datetime.strptime(self.pf.basins[0].header.attrs["Last Modified Date"], "%d %B %Y")
        time = datetime.strptime(self.pf.basins[0].header.attrs["Last Modified Time"], "%H:%M:%S").time()
        return datetime.combine(date, time)

    def _check_files_exists(self, files: list[str]):
        """Ensure the files exists. If they don't raise an error."""
        for file in files:
            if not os.path.exists(file):
                logger.warning(f"File not found {file}")

    def make_thumbnails(self, basins: list[BasinFile], pm: LocalPathManager, overwrite: bool = False) -> list[str]:
        """Create a PNG thumbnail for each basin."""
        thumbnail_paths = []

        for bf in basins:
            thumbnail_path = pm.derived_item_asset(f"{bf.name}.png".replace(" ", "_").replace("-", "_"))

            if not overwrite and os.path.exists(thumbnail_path):
                logger.info(f"Thumbnail for basin `{bf.name}` already exists. Skipping creation.")
            else:
                logger.info(f"{'Overwriting' if overwrite else 'Creating'} thumbnail for basin `{bf.name}`")
                fig = self.make_thumbnail(self, gdfs=bf.hms_schematic_2_gdfs)
                fig.savefig(thumbnail_path)
                fig.clf()
            thumbnail_paths.append(thumbnail_path)

        return thumbnail_paths

    def write_element_geojsons(self, basins: list[BasinFile], pm: LocalPathManager, pf, overwrite: bool = False):
        """Write the HMS elements (Subbasins, Juctions, Reaches, etc.) to geojson."""
        geojson_paths = []
        for element_type in basins[0].elements.element_types:
            # logger.debug(f"Checking if geojson for {element_type} exists")
            path = pm.derived_item_asset(f"{element_type}.geojson")
            if not overwrite and os.path.exists(path):
                logger.info(f"Geojson for {element_type} already exists. Skipping creation.")
            else:
                logger.info(f"Creating geojson for {element_type}")
                gdf = pf.basins[0].feature_2_gdf(element_type).to_crs(4326)
                # logger.debug(gdf.columns)
                keep_columns = ["name", "geometry", "Last Modified Date", "Last Modified Time", "Number Subreaches"]
                gdf = gdf[[col for col in keep_columns if col in gdf.columns]]
                gdf.to_file(path)
            geojson_paths.append(path)

        return geojson_paths

    def add_asset(self, key, asset):
        """Subclass asset then add."""
        subclass = self.factory.asset_from_dict(asset)
        if subclass is None:
            return
        return super().add_asset(key, subclass)

    def make_thumbnail(self, gdfs: dict):
        """Create a png from the geodataframes (values of the dictionary). The dictionary keys are used to label the layers in the legend."""
        cdict = {
            "Subbasin": "black",
            "Reach": "blue",
            "Junction": "red",
            "Source": "black",
            "Sink": "green",
            "Reservoir": "cyan",
            "Diversion": "black",
        }
        crs = gdfs["Subbasin"].crs
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Add data
        for layer in gdfs.keys():
            if layer in cdict.keys():
                if layer == "Subbasin":
                    gdfs[layer].plot(ax=ax, edgecolor=cdict[layer], linewidth=1, label=layer, facecolor="none")
                elif layer == "Junction":
                    gdfs[layer].plot(ax=ax, color=cdict[layer], label=layer, markersize=25)
                else:
                    gdfs[layer].plot(ax=ax, color=cdict[layer], linewidth=1, label=layer, markersize=5)
        try:
            ctx.add_basemap(ax, crs=crs, source=ctx.providers.USGS.USTopo)
        except requests.exceptions.HTTPError:
            try:
                ctx.add_basemap(ax, crs=crs, source=ctx.providers.Esri.WorldStreetMap)
            except requests.exceptions.HTTPError:
                ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenStreetMap.Mapnik)

        # Format
        # ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    ### Prevent external modification of dynamically generated properties ###
    @geometry.setter
    def geometry(self, *args, **kwargs):
        pass

    @bbox.setter
    def bbox(self, *args, **kwargs):
        pass

    @datetime.setter
    def datetime(self, *args, **kwargs):
        pass
