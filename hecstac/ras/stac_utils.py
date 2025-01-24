import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pystac import Asset, MediaType

from .assets import (
    GeometryAsset,
    GeometryHdfAsset,
    PlanAsset,
    PlanHdfAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    RasAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
)
from .errors import GeometryAssetMissingCRSError, GeometryAssetNoXSError
from .utils import is_ras_prj


@dataclass
class AssetLabel:
    roles: list[str | MediaType]
    description: str


def create_prj_asset(filepath: str) -> ProjectAsset | None:
    if is_ras_prj(filepath):
        return ProjectAsset(filepath)
    return None


def asset_factory(filepath: str, crs: str | None = None, defer_computing_properties: bool = False) -> RasAsset | Asset:
    # check if file is geometry asset, then check that crs is provided
    geometry_asset_pattern = re.compile(r".+\.g\d{2}$", re.IGNORECASE)
    if geometry_asset_pattern.match(filepath):
        if not crs:
            raise GeometryAssetMissingCRSError
        logging.info(f"Creating asset for {filepath} using constructor {GeometryAsset}")
        asset = GeometryAsset(filepath, crs)
        if not defer_computing_properties:
            try:
                asset.populate()
            except GeometryAssetNoXSError:
                asset = Asset(
                    filepath,
                    Path(filepath).name,
                    "The geometry file which contains geometry data, lacking cross-sectional data required to infer spatial location",
                    MediaType.TEXT,
                    ["ras-file"],
                )
        return asset
    # associate file patterns with ras asset constructors
    pattern_ras_constructor_dict: dict[re.Pattern, Callable[[str], RasAsset | None]] = {
        re.compile(r".+\.prj$", re.IGNORECASE): create_prj_asset,
        re.compile(r".+\.p\d{2}$", re.IGNORECASE): PlanAsset,
        re.compile(r".+\.p\d{2}\.hdf$", re.IGNORECASE): PlanHdfAsset,
        re.compile(r".+\.g\d{2}\.hdf$", re.IGNORECASE): GeometryHdfAsset,
        re.compile(r".+\.f\d{2}$", re.IGNORECASE): SteadyFlowAsset,
        re.compile(r".+\.q\d{2}$", re.IGNORECASE): QuasiUnsteadyFlowAsset,
        re.compile(r".+\.u\d{2}$", re.IGNORECASE): UnsteadyFlowAsset,
    }
    for pattern, constructor in pattern_ras_constructor_dict.items():
        if pattern.match(filepath):
            logging.info(f"Creating asset for {filepath} using constructor {constructor}")
            asset = constructor(filepath)
            if asset:
                if not defer_computing_properties:
                    asset.populate()
                return asset
    # associate file patterns with asset label information
    pattern_asset_label_dict: dict[re.Pattern, AssetLabel] = {
        re.compile(r".+\.r\d{2}$", re.IGNORECASE): AssetLabel(
            ["run-file", "ras-file", MediaType.TEXT],
            "Run file for steady flow analysis which contains all the necessary input data required for the RAS computational engine.",
        ),
        re.compile(r".+\.hyd\d{2}$"): AssetLabel(
            ["computational-level-output-file", "ras-file", MediaType.TEXT], "Detailed Computational Level output file."
        ),
        re.compile(r".+\.c\d{2}$", re.IGNORECASE): AssetLabel(
            ["geometric-preprocessor", "ras-file", MediaType.TEXT],
            "Geomatric Pre-Processor output file. Contains the hydraulic properties tables, rating curves, and family of rating curves for each cross-section, bridge, culvert, storage area, inline and lateral structure.",
        ),
        re.compile(r".+\.b\d{2}$", re.IGNORECASE): AssetLabel(
            ["boundary-condition-file", "ras-file", MediaType.TEXT], "Boundary Condition file."
        ),
        re.compile(r".+\.bco\d{2}$"): AssetLabel(
            ["unsteady-flow-log-file", "ras-file", MediaType.TEXT], "Unsteady Flow Log output file."
        ),
        re.compile(r".+\.s\d{2}$", re.IGNORECASE): AssetLabel(
            ["sediment-data-file", "ras-file", MediaType.TEXT],
            "Sediment data file which contains flow data, boundary conditions, and sediment data.",
        ),
        re.compile(r".+\.h\d{2}$", re.IGNORECASE): AssetLabel(
            ["hydraulic-design-file", "ras-file", MediaType.TEXT], "Hydraulic Design data file."
        ),
        re.compile(r".+\.w\d{2}$", re.IGNORECASE): AssetLabel(
            ["water-quality-file", "ras-file", MediaType.TEXT],
            "Water Quality data file which contains temperature boundary conditions, initial conditions, advection dispersion parameters and meteorological data.",
        ),
        re.compile(r".+\.SedCap\d{2}$"): AssetLabel(
            ["sediment-transport-capacity-file", "ras-file", MediaType.TEXT], "Sediment Transport Capacity data."
        ),
        re.compile(r".+\.SedXS\d{2}$"): AssetLabel(
            ["xs-output-file", "ras-file", MediaType.TEXT], "Cross section output file."
        ),
        re.compile(r".+\.SedHeadXS\d{2}$"): AssetLabel(
            ["xs-output-header-file", "ras-file", MediaType.TEXT], "Header file for the cross section output."
        ),
        re.compile(r".+\.wqrst\d{2}$"): AssetLabel(
            ["water-quality-restart-file", "ras-file", MediaType.TEXT], "The water quality restart file."
        ),
        re.compile(r".+\.sed$"): AssetLabel(
            ["sediment-output-file", "ras-file", MediaType.TEXT], "Detailed sediment output file."
        ),
        re.compile(r".+\.blf$"): AssetLabel(["binary-log-file", "ras-file", MediaType.TEXT], "Binary Log file."),
        re.compile(r".+\.dss$"): AssetLabel(
            ["ras-dss", "ras-file"], "The dss file contains the dss results and other simulation information."
        ),
        re.compile(r".+\.log$"): AssetLabel(
            ["ras-log", "ras-file", MediaType.TEXT],
            "The log file contains the log information and other simulation information.",
        ),
        re.compile(r".+\.rst$"): AssetLabel(["restart-file", "ras-file", MediaType.TEXT], "Restart file."),
        re.compile(r".+\.SiamInput$"): AssetLabel(
            ["siam-input-file", "ras-file", MediaType.TEXT], "SIAM Input Data file."
        ),
        re.compile(r".+\.SiamOutput$"): AssetLabel(
            ["siam-output-file", "ras-file", MediaType.TEXT], "SIAM Output Data file."
        ),
        re.compile(r".+\.bco$"): AssetLabel(
            ["water-quality-log", "ras-file", MediaType.TEXT], "Water quality log file."
        ),
        re.compile(r".+\.color-scales$"): AssetLabel(
            ["color-scales", "ras-file", MediaType.TEXT], "File that contains the water quality color scale."
        ),
        re.compile(r".+\.comp-msgs.txt$"): AssetLabel(
            ["computational-message-file", "ras-file", MediaType.TEXT],
            "Computational Message text file which contains the computational messages that pop up in the computation window.",
        ),
        re.compile(r".+\.x\d{2}$", re.IGNORECASE): AssetLabel(
            ["run-file", "ras-file", MediaType.TEXT], "Run file for Unsteady Flow."
        ),
        re.compile(r".+\.o\d{2}$", re.IGNORECASE): AssetLabel(
            ["output-file", "ras-file", MediaType.TEXT], "Output ras file which contains all of the computed results."
        ),
        re.compile(r".+\.IC\.O\d{2}"): AssetLabel(
            ["initial-conditions-file", "ras-file", MediaType.TEXT], "Initial conditions file for unsteady flow plan."
        ),
        re.compile(r".+\.p\d{2}\.rst$"): AssetLabel(["restart-file", "ras-file", MediaType.TEXT], "Restart file."),
        re.compile(r".+\.rasmap$"): AssetLabel(["ras-mapper-file", "ras-file", MediaType.TEXT], "Ras Mapper file."),
        re.compile(r".+\.rasmap\.backup$"): AssetLabel(
            ["ras-mapper-file", "ras-file", MediaType.TEXT], "Backup Ras Mapper file."
        ),
        re.compile(r".+\.rasmap\.original$"): AssetLabel(
            ["ras-mapper-file", "ras-file", MediaType.TEXT], "Original Ras Mapper file."
        ),
        re.compile(r".+\.txt$"): AssetLabel([MediaType.TEXT], "Miscellaneous text file."),
        re.compile(r".+\.xml$"): AssetLabel([MediaType.XML], "Miscellaneous xml file."),
    }
    for pattern, asset_label_metadata in pattern_asset_label_dict.items():
        if pattern.match(filepath):
            asset = Asset(
                filepath,
                title=Path(filepath).name,
                description=asset_label_metadata.description,
                roles=asset_label_metadata.roles,
            )
            return asset
    asset = Asset(filepath, title=Path(filepath).name)
    return asset
