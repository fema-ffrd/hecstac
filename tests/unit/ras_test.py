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

DATA_DIR = Path(__file__).parent.parent / "test_data" / "input" / "ras_test"
OUTPUT_DIR = Path(__file__).parent.parent / "test_data" / "output" / "ras_test"
DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def ras_models():
    directory = DATA_DIR / "metadata.json"
    with open(directory) as f:
        models = json.load(f)
    for m in models:
        prj_path = DATA_DIR / m["directory"] / m["prj_file"]
        yield (str(prj_path), m["crs"])


@pytest.mark.parametrize("prj_path, crs", ras_models())
def test_stac_creation(prj_path: str, crs: str):
    """Test STAC item creation and serialization/deserialization."""
    item = RASModelItem.from_prj(prj_path, crs)

    # In-memory check
    dict_1 = item.to_dict()
    dict_2 = RASModelItem.from_dict(dict_1).to_dict()
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["bbox"]:  # allow only dt diffs
            raise RuntimeError(f"Serialization failed for {prj_path}. The following fields do not match: {bad_fields}")

    # To file check
    out_dir = Path(prj_path.replace(str(DATA_DIR), str(OUTPUT_DIR)))
    out_dir.parent.mkdir(exist_ok=True, parents=True)
    out_path = str(out_dir).replace(".prj", ".json")
    with open(out_path, "w") as f:
        json.dump(dict_1, f, indent=4)
    with open(out_path) as f:
        dict_2 = json.load(f)
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["bbox"]:  # allow only dt diffs
            raise RuntimeError(f"Serialization failed for {prj_path}. The following fields do not match: {bad_fields}")
    print(f"{prj_path} passed")


def dict_comparer(dict_1, dict_2, tb=""):
    bad_fields = []
    for i in dict_1:
        if isinstance(dict_1[i], dict):
            if not isinstance(dict_2[i], dict):
                print_mismatch(i, dict_1, dict_2, tb)
            else:
                tmp_fields = dict_comparer(dict_1[i], dict_2[i], f"{tb}-{i}")
                bad_fields.extend(tmp_fields)
        elif not dict_1[i] == dict_2[i]:
            print_mismatch(i, dict_1, dict_2, f"{tb}-{i}")
            bad_fields.append(i)
    return bad_fields


def print_mismatch(i, dict_1, dict_2, tb):
    print(f"mismatch in {tb}")
    print(dict_1[i])
    print()
    print(dict_2[i])
    print("=" * 50)


@pytest.mark.parametrize("prj_path, crs", ras_models())
def test_thumbnail_creation(prj_path: str, crs: str):
    """Test thumbnail writing for RAS items."""
    # Load item
    item = RASModelItem.from_prj(prj_path, crs)

    # Establish files that will be made and clear if necessary
    out_paths = set()
    out_dir = Path(prj_path.replace(str(DATA_DIR), str(OUTPUT_DIR))).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    for i in item.geometry_assets:
        tmp_path = out_dir / f"thumbnail.{i.name.replace('.hdf', '').split('.')[-1]}.png"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out_paths.add(tmp_path)

    # Create thumbnails
    item.add_model_thumbnails(layers=["XS", "River", "Structure", "Junction", "mesh_areas"], thumbnail_dir=str(out_dir))

    # Check that they were generated
    for i in out_paths:
        assert os.path.exists(i), f"Failed to generate thumbnail for {i}"


@pytest.mark.parametrize("prj_path, crs", ras_models())
def test_geopackage_creation(prj_path: str, crs: str):
    """Test geopackage writing for RAS items."""
    # Load item
    item = RASModelItem.from_prj(prj_path, crs)

    # Establish files that will be made and clear if necessary
    out_paths = set()
    out_dir = Path(prj_path.replace(str(DATA_DIR), str(OUTPUT_DIR))).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    for i in item.geometry_assets:
        tmp_path = out_dir / f"{i.name.replace('.hdf', '')}.gpkg"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        out_paths.add(tmp_path)

    # Create thumbnails
    try:
        item.add_model_geopackages(local_dst=str(out_dir))
    except Invalid1DGeometryError:
        return  # Properly handled

    # Check that they were generated
    for i in out_paths:
        assert os.path.exists(i), f"Failed to generate geopackage for {i}"


if __name__ == "__main__":
    for m in ras_models():
        test_stac_creation(*m)
        test_thumbnail_creation(*m)
        test_geopackage_creation(*m)
