import re
import io
from pathlib import Path
import json
from typing import Tuple
from urllib.parse import urlparse
from hecstac.ras.item import RASModelItem
from hecstac.common.s3_utils import list_keys_regex, init_s3_resources, save_bytes_s3, make_uri_public
from hecstac.common.logger import initialize_logger


def list_calibration_model_files(bucket: str, prefix: str) -> list:
    """List all model files in a given prefix, excluding uNN.hdf and pNN.hdf files."""
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    ras_files = list_keys_regex(s3_client=s3_client, bucket=bucket, prefix_includes=prefix, recursive=False)
    ras_files = [f"s3://{bucket}/{f}" for f in ras_files]
    pattern = re.compile(r"(u|p)\d{2}\.hdf$")
    return [file for file in ras_files if not pattern.search(file)]


def parse_ras_project_path(s3_path: str) -> Tuple[str, str, str]:
    """Parse an S3 path to extract the bucket, prefix, and RAS model name."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    path = Path(key)

    ras_model_name = path.stem
    prefix = str(path.parent)

    return bucket, prefix, ras_model_name


def create_calibration_item(ras_project_path: str, output_prefix: str):
    """
    Generate and upload a STAC item for a RAS model calibration run.

    Args:
        ras_project_path (str): S3 path to the RAS project file.
        output_prefix (str): S3 prefix for storing the output item and assets.

    """
    bucket, prefix, ras_model_name = parse_ras_project_path(ras_project_path)

    output_item_prefix = f"{output_prefix}/model={ras_model_name}"
    output_item_path = f"{output_item_prefix}/item.json"
    output_assets_prefix = f"{output_prefix}/model={ras_model_name}/data=geometry"

    calibration_model_files = list_calibration_model_files(bucket, prefix)

    ras_item = RASModelItem.from_prj(ras_project_path, crs=None, assets=calibration_model_files)
    ras_item.set_self_href(make_uri_public(output_item_path))
    ras_item.add_model_thumbnails(layers=["mesh_areas", "breaklines", "bc_lines"], s3_thumbnail_dir=output_item_prefix)
    ras_item.add_geospatial_assets(output_assets_prefix)

    item_dict = ras_item.to_dict()
    item_bytes = io.BytesIO(json.dumps(item_dict, indent=2).encode("utf-8"))

    save_bytes_s3(data=item_bytes, s3_path=output_item_path, content_type="application/json")


if __name__ == "__main__":
    _, s3_client, _ = init_s3_resources()
    initialize_logger()

    config = {
        "ras_project_path": f"s3://trinity-pilot/calibration/hydraulics/bedias-creek/bedias-creek.prj",
        "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration",
    }

    create_calibration_item(config["ras_project_path"], config["output_prefix"])
