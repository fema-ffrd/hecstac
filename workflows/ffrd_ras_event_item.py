"""Creates STAC Item and adds time series assets representing an HEC-RAS FFRD event."""

import io
import json
import re
import argparse
from rashdf import RasPlanHdf
from openpyxl.utils.exceptions import IllegalCharacterError
from dotenv import load_dotenv
from typing import Optional

from hecstac.common.logger import initialize_logger
from hecstac.common.utils import load_config
from hecstac.common.s3_utils import init_s3_resources, list_keys_regex, parse_s3_url, save_bytes_s3
from hecstac.events.ras_ffrd import FFRDEventItem


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Requires:
        --config: Path to a JSON configuration file or JSON string.

    Example JSON config:

        Single:
        '{
            "model_prefix": "s3://trinity-pilot/calibration/hydraulics/bedias-creek",
            "source_model_path": "s3://trinity-pilot/stac/prod-support/calibration/model=bedias-creek/item.json",
            "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration"
        }'

        List:
        '[
            {
                "model_prefix": "s3://trinity-pilot/calibration/hydraulics/bedias-creek",
                "source_model_path": "s3://trinity-pilot/stac/prod-support/calibration/model=bedias-creek/item.json",
                "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration"
            },
            {
                "model_prefix": "s3://trinity-pilot/calibration/hydraulics/blw-bear",
                "source_model_path": "s3://trinity-pilot/stac/prod-support/calibration/model=blw-bear/item.json",
                "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration"
            }
        ]'

    """
    parser = argparse.ArgumentParser(description="Create STAC Item(s) for HEC-RAS model events on S3.")
    parser.add_argument("--config", required=True, help="JSON string or path to JSON config file.")
    return parser.parse_args()


def extract_plan_info(plan_path: str) -> tuple[str, str]:
    """Extract model name and event name from given plan file."""
    plan_hdf = RasPlanHdf.open_uri(plan_path)

    model_name = plan_path.split("/")[-1].split(".")[0]
    event_name = plan_hdf.get_plan_info_attrs()["Plan Name"]

    return model_name, event_name


def list_plan_hdfs(s3_client, model_prefix: str) -> list[str]:
    """List all plan HDF files in the given S3 prefix."""
    bucket, prefix = parse_s3_url(model_prefix)
    ras_files = list_keys_regex(s3_client=s3_client, bucket=bucket, prefix_includes=prefix)
    ras_files = [f"s3://{bucket}/{f}" for f in ras_files]
    plan_hdf_files = [f for f in ras_files if re.search(r"\.p\d{2}\.hdf$", f)]
    if plan_hdf_files:
        return plan_hdf_files
    else:
        raise ValueError(f"No plan hdf files found at bucket: {bucket} and prefix: {prefix} ")


def create_event_item(
    plan_file_path: str, source_model_path: str, output_prefix: str, calibration_only: bool = False
) -> Optional[FFRDEventItem]:
    """
    Create and upload a STAC item for a RAS model event.

    Args:
        plan_file_path (str): Path to the RAS plan file (.pXX).
        source_model_path (str): Path to the source model used for the event.
        output_prefix (str): S3 prefix for storing the output item and assets.
        calibration_only (bool): If True, only process events with 'calibration' in the name.

    Returns
    -------
        FFRDEventItem: The created FFRD event item.
    """
    model_name, event_name = extract_plan_info(plan_file_path)

    if calibration_only and "calibration" not in event_name:
        logger.warning(f"{event_name} does not contain 'calibration', skipping...")
        return None

    logger.info(f"Creating stac item for event: {event_name}")

    short_event_name = event_name.split("_")[1] if calibration_only else event_name

    assets_prefix = f"{output_prefix}/model={model_name}/event={short_event_name}"
    dest_href = f"{assets_prefix}/item.json"

    event_item = FFRDEventItem(
        ras_simulation_files=[plan_file_path], source_model_paths=[source_model_path], event_id=short_event_name
    )

    event_item.add_ts_assets(assets_prefix)
    event_item.set_self_href(dest_href)
    event_item.validate()
    return event_item


def main(config) -> None:
    """
    Create STAC Items and time series assets for RAS event model.

    Args:
        config (dict): Configuration dictionary loaded from a JSON string.
    """
    try:
        _, s3_client, _ = init_s3_resources()
        plan_hdfs = list_plan_hdfs(s3_client, config["model_prefix"])

        for plan_hdf in plan_hdfs:
            event_item = create_event_item(
                plan_file_path=plan_hdf,
                source_model_path=config["source_model_path"],
                output_prefix=config["output_prefix"],
            )
            item_dict = event_item.to_dict()
            item_bytes = io.BytesIO(json.dumps(item_dict, indent=2).encode("utf-8"))

            save_bytes_s3(data=item_bytes, s3_path=event_item.self_href, content_type="application/json")
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
        except ValueError as e:
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
