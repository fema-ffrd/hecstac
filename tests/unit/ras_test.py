import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from shapely import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.geometry import shape

from hecstac.common.logger import initialize_logger
from hecstac.common.base_io import ModelFileReaderError
from hecstac.ras.errors import Invalid1DGeometryError
from hecstac.ras.item import RASModelItem
from hecstac.common.consts import OSM_ATTRIBUTION, OSM_USER_AGENT
from hecstac.ras.utils import export_thumbnail, handle_spaces, search_contents
from hecstac.ras.parser import GeometryFile, ProjectFile, Structure, StructureType

initialize_logger(level=logging.CRITICAL)

DATA_DIR = Path(__file__).parent.parent / "test_data" / "input" / "ras_test"
OUTPUT_DIR = Path(__file__).parent.parent / "test_data" / "output" / "ras_test"
DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def ras_models():
    directory = DATA_DIR / "metadata.json"
    with open(directory) as f:
        meta = json.load(f)
    for m in meta["models"]:
        yield (m["prj_path"], m["crs"], m["assets"])


@pytest.mark.parametrize("prj_path, crs, assets", ras_models())
def test_stac_creation(prj_path: str, crs: str, assets: list):
    """Test STAC item creation and serialization/deserialization."""
    item = RASModelItem.from_prj(prj_path, crs, assets=assets)

    # In-memory check
    dict_1 = item.to_dict()
    dict_2 = RASModelItem.from_dict(dict_1).to_dict()
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["bbox"]:  # allow only dt diffs
            raise RuntimeError(f"Serialization failed for {prj_path}. The following fields do not match: {bad_fields}")

    # To file check
    out_dir = Path(OUTPUT_DIR) / Path(prj_path).parent.name
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = str(out_dir / f"{Path(prj_path).parent.name}.stac.json")
    with open(out_path, "w") as f:
        json.dump(dict_1, f, indent=4)
    with open(out_path) as f:
        dict_2 = json.load(f)
    if dict_1 != dict_2:
        bad_fields = dict_comparer(dict_1, dict_2)
        if bad_fields != ["bbox"]:  # allow only dt diffs
            raise RuntimeError(f"Serialization failed for {prj_path}. The following fields do not match: {bad_fields}")
    print(f"{prj_path} passed")


def test_unlocated_item_uses_null_geometry_without_bbox():
    """Serialize a model without a CRS as an unlocated STAC item."""
    item = RASModelItem("unlocated", None, None, datetime.now(timezone.utc), {}, assets={})

    item_dict = item.to_dict()

    assert item_dict["geometry"] is None
    assert "bbox" not in item_dict


@pytest.mark.parametrize(
    "source_geometry",
    [
        GeometryCollection(
            [
                Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
                LineString([(20, 20), (21, 21)]),
            ]
        ),
        MultiPolygon(
            [
                Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
                Polygon([(10, 0), (11, 0), (11, 1), (10, 1)]),
            ]
        ),
    ],
)
def test_geometry_returns_largest_polygonal_component(source_geometry):
    item = RASModelItem("geometry-test", None, None, datetime.now(timezone.utc), {}, assets={})
    item.crs = "EPSG:4326"
    item.geometry_assets = [SimpleNamespace(href="geometry.g01", geometry_wgs84=source_geometry)]

    geometry = shape(item.geometry)

    assert geometry.geom_type == "Polygon"
    assert geometry.area == 16


def test_search_contents_preserves_token_in_value():
    assert search_contents(["key=value=tail"], "key") == "value=tail"


def test_handle_spaces_rejects_missing_variant():
    with pytest.raises(ValueError, match="not found in lines"):
        handle_spaces("missing=value", ["other=value"])


def test_export_thumbnail_uses_identified_osm_basemap(tmp_path, monkeypatch):
    basemap_options = {}

    def capture_basemap(ax, **kwargs):
        basemap_options.update(kwargs)

    monkeypatch.setattr("hecstac.ras.utils.ctx.add_basemap", capture_basemap)
    thumbnail_path = tmp_path / "thumbnail.png"

    export_thumbnail([], "thumbnail", "EPSG:4326", str(thumbnail_path))

    assert basemap_options["source"]["url"] == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    assert basemap_options["headers"]["user-agent"] == OSM_USER_AGENT
    assert basemap_options["attribution"] == OSM_ATTRIBUTION
    assert thumbnail_path.is_file()


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


@pytest.mark.parametrize("prj_path, crs, assets", ras_models())
def test_thumbnail_creation(prj_path: str, crs: str, assets: list):
    """Test thumbnail writing for RAS items."""
    # Load item
    item = RASModelItem.from_prj(prj_path, crs, assets=assets)

    # Establish files that will be made and clear if necessary
    out_dir = Path(OUTPUT_DIR) / Path(prj_path).parent.name
    out_dir.mkdir(exist_ok=True, parents=True)

    # Create thumbnails
    item.add_model_thumbnails(
        layers=["XS", "River", "Structure", "Junction", "mesh_areas"], thumbnail_dest=str(out_dir)
    )

    # Check that they were generated
    for i in [i.href for i in item.assets.values() if i.roles and "thumbnail" in i.roles]:
        assert os.path.exists(i), f"Failed to generate thumbnail for {i}"


@pytest.mark.parametrize("prj_path, crs, assets", ras_models())
def test_geopackage_creation(prj_path: str, crs: str, assets: list):
    """Test geopackage writing for RAS items."""
    # Load item
    item = RASModelItem.from_prj(prj_path, crs, assets=assets)

    # Establish files that will be made and clear if necessary
    out_dir = Path(OUTPUT_DIR) / Path(prj_path).parent.name
    out_dir.mkdir(exist_ok=True, parents=True)

    # Create thumbnails
    try:
        item.add_model_geopackages(dst=str(out_dir))
    except Invalid1DGeometryError:
        return  # Properly handled

    # Check that they were generated
    for i in [i.href for i in item.assets.values() if i.roles and "RAS-GEOMETRY-GPKG" in i.roles]:
        assert os.path.exists(i), f"Failed to generate geopackage for {i}"


# Test for structure width and distance parsing from geometry files
@pytest.fixture(scope="module")
def geometry_files() -> list[GeometryFile]:
    """Resolve the project's geometry files and parse them."""
    try:
        prj_path = get_baxter_prj_from_metadata()
        prj = ProjectFile(prj_path)
        assert prj.geometry_files, f"No geometry files listed in {prj_path}"
        return [GeometryFile(associated_file_path(prj_path, g)) for g in prj.geometry_files]
    except ModelFileReaderError as e:
        pytest.skip(f"Unable to read RAS project {prj_path}: {e}")
    return []  # Return an empty list if skipped to satisfy the return type


def get_baxter_prj_from_metadata() -> str:
    """Return the Baxter .prj path listed in test_data/input/ras_test/metadata.json."""
    ras_prj_json = DATA_DIR / "metadata.json"
    with open(ras_prj_json) as f:
        models = json.load(f)["models"]
    return next(m["prj_path"] for m in models if m["prj_path"].endswith("/Baxter.prj"))


def associated_file_path(path: str, name: str) -> str:
    """Return a file path in the same directory as path; works for S3 URIs and local paths."""
    return path[: max(path.rfind("/"), path.rfind("\\")) + 1] + name


def test_structure_width_and_distance(geometry_files: list[GeometryFile]):
    """Test that the parsed structure width and distance match the values from the geometry file."""
    for geom in geometry_files:  # Using test case Baxter.prj from ffrd-pilot. There is only 1 geometry file
        # Check the first structure information against the expected values
        first_key, first_val = next(iter(geom.structures.items()), (None, None))
        if first_key:
            parsed = {
                (first_val.river.strip(), first_val.reach.strip(), first_val.river_station): (
                    first_val.distance,
                    first_val.width,
                )
            }

        expected = {("Baxter River", "Upper Reach", 77737.0): (140, 20)}

        assert parsed.keys() == expected.keys(), f"{geom.fpath}: structure mismatch"
        for key, (distance, width) in parsed.items():
            assert (distance, width) == expected[key], (
                f"{geom.fpath} {key}: got {(distance, width)}, expected {expected[key]}"
            )


# Unit tests using synthetic RAS data for distance and width of structures.
def make_structure(ras_data: list[str]) -> Structure:
    return Structure(ras_data, "River,Reach", "River", "Reach", us_xs=None)


@pytest.mark.parametrize(
    "structure_type_code, structure_type",
    [
        (2, StructureType.CULVERT),
        (3, StructureType.BRIDGE),
        (4, StructureType.MULTIPLE_OPENING),
    ],
)
def test_width_deck_structures(structure_type_code, structure_type):
    ras_data = [
        f"Type RM Length L Ch R = {structure_type_code} ,1050    ,,,",
        "Deck Dist Width WeirC Skew NumUp NumDn MinLoCord MaxHiCord MaxSubmerge Is_Ogee",
        " 25,40.5,2.6,0,10,10,,,.95,0,",
        "Deck Dist Width End",
    ]
    structure = make_structure(ras_data)

    assert structure.type == structure_type
    assert structure.width == 40.5
    assert structure.distance == 25.0


def test_width_inline_structure():
    ras_data = [
        "Type RM Length L Ch R = 5 ,1050    ,,,",
        "IW Dist,WD,Coef,Skew,MaxSub,Min_El,Is_Ogee,SpillHt,DesHd",
        " 50,20,2.6,0,.95,,0,,",
        "IW End",
    ]
    structure = make_structure(ras_data)

    assert structure.type == StructureType.INLINE_STRUCTURE
    assert structure.width == 20.0
    assert structure.distance == 50.0


def test_width_lateral_structure():
    ras_data = ["Type RM Length L Ch R = 6 ,1050    ,,,"]
    structure = make_structure(ras_data)

    assert structure.type == StructureType.LATERAL_STRUCTURE
    assert structure.width == 0.0


if __name__ == "__main__":
    for m in ras_models():
        test_stac_creation(*m)
        test_thumbnail_creation(*m)
        test_geopackage_creation(*m)
