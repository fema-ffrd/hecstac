import json
import logging
import os
from datetime import datetime
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import requests
from pystac import Item
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.storage import StorageExtension
from shapely import to_geojson, union_all

from hecstac.common.asset_factory import AssetFactory
from hecstac.common.path_manager import LocalPathManager
from hecstac.hms.assets import HMS_EXTENSION_MAPPING, ProjectAsset
from hecstac.hms.parser import BasinFile, ProjectFile


class HMSModelItem(Item):
    """An object representation of a HEC-HMS model."""

    PROJECT = "hms:project"
    PROJECT_TITLE = "hms:project_title"
    MODEL_UNITS = "hms:unit system"
    MODEL_GAGES = "hms:gages"
    PROJECT_VERSION = "hms:version"
    PROJECT_DESCRIPTION = "hms:description"
    PROJECT_UNITS = "hms:unit_system"

    def __init__(self, hms_project_file, item_id: str, simplify_geometry: bool = True):

        self._project = None
        self.assets = {}
        self.links = []
        self.thumbnail_paths = []
        self.geojson_paths = []
        self.extra_fields = {}
        self.stac_extensions = None
        self.pm = LocalPathManager(Path(hms_project_file).parent)
        self._href = self.pm.item_path(item_id)
        self.hms_project_file = hms_project_file
        self._simplify_geometry = simplify_geometry

        self.pf = ProjectFile(self.hms_project_file, assert_uniform_version=False)
        self.factory = AssetFactory(HMS_EXTENSION_MAPPING)

        super().__init__(
            Path(self.hms_project_file).stem,
            self._geometry,
            self._bbox,
            self._datetime,
            self._properties,
            href=self._href,
        )

        self._check_files_exists(self.pf.files + self.pf.rasters)
        self.make_thumbnails(self.pf.basins)
        self.write_element_geojsons(self.pf.basins[0])
        for fpath in self.thumbnail_paths + self.geojson_paths + self.pf.files + self.pf.rasters:
            self.add_hms_asset(fpath)

        self._register_extensions()

    def _register_extensions(self) -> None:
        ProjectionExtension.add_to(self)
        StorageExtension.add_to(self)

    @property
    def _properties(self):
        """Properties for the HMS STAC item."""
        properties = {}
        properties[self.PROJECT] = f"{self.pf.name}.hms"
        properties[self.PROJECT_TITLE] = self.pf.name
        properties[self.PROJECT_VERSION] = (self.pf.attrs["Version"],)
        properties[self.PROJECT_DESCRIPTION] = (self.pf.attrs.get("Description"),)

        # TODO probably fine 99% of the time but we grab this info from the first basin file only
        properties[self.MODEL_UNITS] = self.pf.basins[0].attrs["Unit System"]
        properties[self.MODEL_GAGES] = self.pf.basins[0].gages

        properties["proj:code"] = self.pf.basins[0].epsg
        if self.pf.basins[0].epsg:
            logging.warning("No EPSG code found in basin file.")
        properties["proj:wkt"] = self.pf.basins[0].wkt
        properties["hms:summary"] = self.pf.file_counts
        return properties

    @property
    def _bbox(self) -> tuple[float, float, float, float]:
        """Bounding box of the HMS STAC item."""
        if len(self.pf.basins) == 0:
            return [0, 0, 0, 0]
        else:
            bboxes = np.array([i.bbox(4326) for i in self.pf.basins])
            bboxes = [bboxes[:, 0].min(), bboxes[:, 1].min(), bboxes[:, 2].max(), bboxes[:, 3].max()]
            return [float(i) for i in bboxes]

    @property
    def _geometry(self) -> dict | None:
        """Geometry of the HMS STAC item. Union of all basins in the HMS model."""
        if self._simplify_geometry:
            geometries = [b.basin_geom.simplify(0.001) for b in self.pf.basins]
        else:
            geometries = [b.basin_geom for b in self.pf.basins]
        return json.loads(to_geojson(union_all(geometries)))

    @property
    def _datetime(self) -> datetime:
        """The datetime for the HMS STAC item."""
        date = datetime.strptime(self.pf.basins[0].header.attrs["Last Modified Date"], "%d %B %Y")
        time = datetime.strptime(self.pf.basins[0].header.attrs["Last Modified Time"], "%H:%M:%S").time()
        return datetime.combine(date, time)

    def _check_files_exists(self, files: list[str]):
        """Ensure the files exists. If they don't rasie an error."""
        from pathlib import Path

        for file in files:
            if not os.path.exists(file):
                logging.warning(f"File not found {file}")

    def make_thumbnails(self, basins: list[BasinFile], overwrite: bool = False):
        """Create a png for each basin. Optionally overwrite existing files."""
        for bf in basins:
            thumbnail_path = self.pm.derived_item_asset(f"{bf.name}.png".replace(" ", "_").replace("-", "_"))

            if not overwrite and os.path.exists(thumbnail_path):
                logging.info(f"Thumbnail for basin `{bf.name}` already exists. Skipping creation.")
            else:
                logging.info(f"{'Overwriting' if overwrite else 'Creating'} thumbnail for basin `{bf.name}`")
                fig = self.make_thumbnail(bf.hms_schematic_2_gdfs)
                fig.savefig(thumbnail_path)
                fig.clf()
            self.thumbnail_paths.append(thumbnail_path)

    def write_element_geojsons(self, basins: list[BasinFile], overwrite: bool = False):
        """Write the HMS elements (Subbasins, Juctions, Reaches, etc.) to geojson."""
        for element_type in basins.elements.element_types:
            logging.debug(f"Checking if geojson for {element_type} exists")
            path = self.pm.derived_item_asset(f"{element_type}.geojson")
            if not overwrite and os.path.exists(path):
                logging.info(f"Geojson for {element_type} already exists. Skipping creation.")
            else:
                logging.info(f"Creating geojson for {element_type}")
                gdf = self.pf.basins[0].feature_2_gdf(element_type).to_crs(4326)
                logging.debug(gdf.columns)
                keep_columns = ["name", "geometry", "Last Modified Date", "Last Modified Time", "Number Subreaches"]
                gdf = gdf[[col for col in keep_columns if col in gdf.columns]]
                gdf.to_file(path)
            self.geojson_paths.append(path)

    def add_hms_asset(self, fpath: str) -> None:
        """Add an asset to the HMS STAC item."""
        if os.path.exists(fpath):
            asset = self.factory.create_hms_asset(fpath)
            if asset is not None:
                self.add_asset(asset.title, asset)
                if isinstance(asset, ProjectAsset):
                    if self._project is not None:
                        logging.error(
                            f"Only one project asset is allowed. Found {str(asset)} when {str(self._project)} was already set."
                        )
                    self._project = asset

    def make_thumbnail(self, gdfs: dict):
        """Create a png from the geodataframes (values of the dictionary).
        The dictionary keys are used to label the layers in the legend."""
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
