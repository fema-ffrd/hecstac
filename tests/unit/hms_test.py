import json
import os
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pytest

from hecstac.common.base_io import ModelFileReaderError
from hecstac.common.s3_utils import download_data_from_s3
from hecstac.hms.item import HMSModelItem
from hecstac.hms.parser import (
    ProjectFile,
    TerrainFile,
    RunFile,
    GridFile,
    GageFile,
    PairedDataFile,
    MetFile,
    BasinFile,
    ControlFile,
)

DATA_DIR = Path(__file__).parent.parent / "test_data" / "input" / "hms_test"
OUTPUT_DIR = Path(__file__).parent.parent / "test_data" / "output" / "hms_test"
DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@pytest.fixture(scope="session")
def local_hms_model():
    """Fixture to download a HMS model from s3 to a local directory for testing."""
    directory = DATA_DIR / "metadata.json"
    with open(directory) as f:
        meta = json.load(f)
    hms_s3_path = meta["models"][1]["prj_path"]
    clean_s3_path = hms_s3_path.replace("s3://", "")
    s3_path_obj = PurePosixPath(clean_s3_path)
    model_dir_name = s3_path_obj.parent.name
    model_name = s3_path_obj.name
    s3_prefix = f"s3://{s3_path_obj.parent}/"

    assets_list = meta["models"][1]["assets"]
    asset_file_names = [PurePosixPath(path.replace("s3://", "")).name for path in assets_list]
    if model_name not in asset_file_names:
        asset_file_names.append(model_name)

    local_download_dir = DATA_DIR / model_dir_name
    download_data_from_s3(s3_prefix, asset_file_names, local_download_dir)
    return local_download_dir, model_name


def test_stac_creation(local_hms_model):
    """Test STAC item creation and serialization/deserialization."""
    local_download_dir, file_name = local_hms_model
    item_id = file_name[:-4] + "-hms-model"
    prj_path = str(local_download_dir / file_name)
    out_dir = Path(OUTPUT_DIR) / local_download_dir.name
    out_dir.mkdir(exist_ok=True, parents=True)
    hms_item = HMSModelItem.from_prj(prj_path, item_id, asset_dir=str(out_dir))

    # In-memory check
    dict_1 = hms_item.to_dict()
    dict_2 = HMSModelItem.from_dict(dict_1).to_dict()
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["bbox"]:  # allow only dt diffs
            raise RuntimeError(
                f"Serialization failed for {local_hms_model}. The following fields do not match: {bad_fields}"
            )

    # To file check
    out_path = str(out_dir / f"{item_id}.stac.json")
    with open(out_path, "w") as f:
        json.dump(dict_1, f, indent=4)
    with open(out_path) as f:
        dict_2 = json.load(f)
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["bbox"]:  # allow only dt diffs
            raise RuntimeError(
                f"Serialization failed for {local_hms_model}. The following fields do not match: {bad_fields}"
            )
    print(f"{local_hms_model} passed")


def dict_comparer(dict_1, dict_2, tb=""):
    bad_fields = []
    for i in dict_1:
        if isinstance(dict_1[i], dict):
            if not isinstance(dict_2[i], dict):
                print_mismatch(i, dict_1, dict_2, tb)
            else:
                tmp_fields = dict_comparer(dict_1[i], dict_2[i], f"{tb}-{i}")
                bad_fields.extend(tmp_fields)
        elif dict_1[i] != dict_2[i]:
            print_mismatch(i, dict_1, dict_2, f"{tb}-{i}")
            bad_fields.append(i)
    return bad_fields


def print_mismatch(i, dict_1, dict_2, tb):
    print(f"mismatch in {tb}")
    print(dict_1[i])
    print()
    print(dict_2[i])
    print("=" * 50)
