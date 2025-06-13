"""Item representing an FFRD item."""

import io
import json
import re

import fsspec
from rashdf import RasPlanHdf

from hecstac.common.logger import initialize_logger
from hecstac.common.s3_utils import init_s3_resources, list_keys_regex, parse_s3_url, save_bytes_s3
from hecstac.events.ffrd import FFRDEventItem


_, s3_client, _ = init_s3_resources()
logger = initialize_logger()


def extract_plan_info(plan_path: str):
    """Extract model name and event name from given plan file."""
    plan_hdf = RasPlanHdf.open_uri(plan_path)

    model_name = plan_path.split("/")[-1].split(".")[0]
    event_name = plan_hdf.get_plan_info_attrs()["Plan Name"]

    return model_name, event_name


def list_plan_hdfs(model_prefix: str) -> list:
    """List all plan HDF files in the given S3 prefix."""
    bucket, prefix = parse_s3_url(model_prefix)
    ras_files = list_keys_regex(s3_client=s3_client, bucket=bucket, prefix_includes=prefix)
    ras_files = [f"s3://{bucket}/{f}" for f in ras_files]
    plan_hdf_files = [f for f in ras_files if re.search(r"\.[p]\d{2}\.hdf$", f)]
    if plan_hdf_files:
        return plan_hdf_files
    else:
        raise ValueError(f"No plan hdf files found at bucket: {bucket} and prefix: {prefix} ")


def create_event_item(plan_file_path: str, source_model_path: str, output_prefix: str, calibration_only: bool = False):
    """
    Create and upload a STAC item for a RAS model event.

    Args:
        plan_file_path (str): Path to the RAS plan file (.pXX).
        source_model_path (str): Path to the source model used for the event.
        output_prefix (str): S3 prefix for storing the output item and assets.
        calibration_only (bool): If True, only process events with 'calibration' in the name.
    """
    model_name, event_name = extract_plan_info(plan_file_path)

    if calibration_only and "calibration" not in event_name:
        logger.warning(f"{event_name} does not contain 'calibration', skipping...")
        return

    logger.info(f"Creating stac item for event: {event_name}")

    short_event_name = event_name.split("_")[1] if calibration_only else event_name

    assets_prefix = f"{output_prefix}/model={model_name}/event={short_event_name}"
    dest_href = f"{assets_prefix}/item.json"

    event_item = FFRDEventItem(
        ras_simulation_files=[plan_file_path], source_model_paths=[source_model_path], event_id=short_event_name
    )

    event_item.add_ts_assets(assets_prefix)
    event_item.validate()

    item_dict = event_item.to_dict()
    item_bytes = io.BytesIO(json.dumps(item_dict, indent=2).encode("utf-8"))

    save_bytes_s3(data=item_bytes, s3_path=dest_href, content_type="application/json")


if __name__ == "__main__":
    config = {
        "model_prefix": "s3://trinity-pilot/calibration/hydraulics/bedias-creek",
        "source_model_path": "s3://trinity-pilot/stac/prod-support/calibration/model=bedias-creek/item.json",
        "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration",
    }
    plan_hdfs = list_plan_hdfs(config["model_prefix"])

    for plan_hdf in plan_hdfs:
        create_event_item(
            plan_file_path=plan_hdf,
            source_model_path=config["source_model_path"],
            output_prefix=config["output_prefix"],
        )
