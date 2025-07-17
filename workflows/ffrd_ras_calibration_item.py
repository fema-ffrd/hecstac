"""Item representing a HEC-RAS FFRD calibration model."""

import io
import json
import re
from pathlib import Path
from typing import Tuple
import argparse
from dotenv import load_dotenv
from openpyxl.utils.exceptions import IllegalCharacterError

from hecstac.common.utils import load_config
from hecstac.common.logger import initialize_logger
from hecstac.common.s3_utils import init_s3_resources, list_keys_regex, make_uri_public, save_bytes_s3, parse_s3_url
from hecstac.ras.item import RASModelItem


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Requires:
        --config: Path to a JSON configuration file or JSON string.

    Example JSON config:

        Single:
        '{
        "ras_project_path": "s3://trinity-pilot/calibration/hydraulics/clear-creek/clear-creek.prj",
        "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration"
        }'

        List:
        '[
            {
                "ras_project_path": "s3://trinity-pilot/calibration/hydraulics/bedias-creek/bedias-creek.prj",
                "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration"
            },
            {
                "ras_project_path": "s3://trinity-pilot/calibration/hydraulics/blw-bear/blw-bear.prj",
                "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration"
            }
        ]'

    """
    parser = argparse.ArgumentParser(description="Create STAC Item(s) for HEC-RAS calibration models on S3.")
    parser.add_argument("--config", required=True, help="JSON string or path to JSON config file.")
    return parser.parse_args()


def list_calibration_model_files(s3_client, bucket: str, prefix: str) -> list[str]:
    """List all model files in a given prefix, excluding uNN.hdf and pNN.hdf files."""
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    ras_files = list_keys_regex(s3_client=s3_client, bucket=bucket, prefix_includes=prefix, recursive=False)
    ras_files = [f"s3://{bucket}/{f}" for f in ras_files]
    pattern = re.compile(r"[up]\d{2}\.hdf$")
    return [file for file in ras_files if not pattern.search(file)]


def parse_ras_project_path(s3_path: str) -> Tuple[str, str, str]:
    """Parse an S3 path to extract the bucket, prefix, and RAS model name."""
    bucket, key = parse_s3_url(s3_path)
    path = Path(key)

    ras_model_name = path.stem
    prefix = str(path.parent)

    return bucket, prefix, ras_model_name


def create_calibration_item(s3_client, ras_project_path: str, output_prefix: str) -> tuple[RASModelItem, str]:
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

    calibration_model_files = list_calibration_model_files(s3_client, bucket, prefix)

    ras_item = RASModelItem.from_prj(ras_project_path, assets=calibration_model_files)
    ras_item.set_self_href(make_uri_public(output_item_path))
    ras_item.add_model_thumbnails(layers=["mesh_areas", "breaklines", "bc_lines"], thumbnail_dest=output_item_prefix)
    ras_item.add_geospatial_assets(output_assets_prefix)
    ras_item.validate()

    return ras_item, output_item_path


def main(config) -> None:
    """
    Create STAC Items and geospatial assets for RAS calibration model.

    Args:
        config (dict): Configuration dictionary loaded from a JSON string.
    """
    try:
        _, s3_client, _ = init_s3_resources()
        calibration_item, output_item_path = create_calibration_item(
            s3_client, ras_project_path=config["ras_project_path"], output_prefix=config["output_prefix"]
        )
        item_dict = calibration_item.to_dict()
        item_bytes = io.BytesIO(json.dumps(item_dict, indent=2).encode("utf-8"))
        save_bytes_s3(data=item_bytes, s3_path=output_item_path, content_type="application/json")

    except IllegalCharacterError as e:
        logger.error(
            f"Unable to create calibration report, please review ras file contents and fix non-ascii characters: {e}"
        )
        raise
    except KeyError as e:
        logger.error(f"Missing required config key: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    logger = initialize_logger()

    try:
        dotenv_loaded = load_dotenv()
        if not dotenv_loaded:
            logger.warning(".env file not found or not loaded.")

        args = parse_args()

        try:
            configs = load_config(args.config)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid config input: {e}")
            raise SystemExit(1)

        for idx, config in enumerate(configs, 1):
            logger.info(f"Processing config {idx}/{len(configs)}: {config}")
            try:
                main(config)
            except Exception as e:
                continue

    except Exception as e:
        logger.exception(f"Fatal error during initialization or processing: {e}")
        raise SystemExit(1)
