import json
import logging
import os
import shutil
from ast import mod
from pathlib import Path
from pyexpat import model

import pytest

from hecstac.common.logger import initialize_logger
from hecstac.ras.errors import Invalid1DGeometryError
from hecstac.ras.item import RASModelItem

initialize_logger(level=logging.CRITICAL)

DATA_DIR = Path(__file__).parent.parent / "test_data" / "input" / "no_asset_test"
OUTPUT_DIR = Path(__file__).parent.parent / "test_data" / "output" / "no_asset_test"
DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_project_w_no_other_assets():
    """Test if a valid stac json is created from a model with just a prj file."""
    directory = DATA_DIR / "metadata.json"
    with open(directory) as f:
        meta = json.load(f)
    item = RASModelItem.from_prj(meta["prj_path"], crs=meta["crs"], assets=meta["assets"])
    item.to_dict()


if __name__ == "__main__":
    test_project_w_no_other_assets()
