import json
import logging
from ast import mod
from pathlib import Path
from pyexpat import model

import pytest

from hecstac.common.logger import initialize_logger
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
    models = [i for i in models if i["directory"] == "0ad5b6c4252c9a1bcf261674527e289fa751eafd981d444c5680fe200fe5e2e3"]
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
        if bad_fields != ["datetime"]:  # allow only dt diffs
            raise RuntimeError(f"Serialization failed for {prj_path}. The following fields do not match: {bad_fields}")

    # To file check
    out_dir = Path(prj_path.replace(str(DATA_DIR), str(OUTPUT_DIR))).parent.mkdir(exist_ok=True, parents=True)
    out_path = str(out_dir).replace(".prj", ".json")
    with open(out_path, "w") as f:
        json.dump(dict_1, f, indent=4)
    with open(out_path) as f:
        dict_2 = json.load(f)
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["datetime"]:  # allow only dt diffs
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


if __name__ == "__main__":
    for m in ras_models():
        test_stac_creation(*m)
        # test_project_file_access(*m)
