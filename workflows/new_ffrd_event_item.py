import re
import io
import fsspec
import json
from rashdf import RasPlanHdf
from hecstac.common.s3_utils import list_keys_regex, init_s3_resources, save_bytes_s3
from hecstac.common.logger import initialize_logger
from hecstac.events.ffrd import FFRDEventItem

fs = fsspec.filesystem("s3")
_, s3_client, _ = init_s3_resources()
logger = initialize_logger()


def extract_plan_info(plan_path):
    """Extract model name and event name from given plan file."""
    plan_hdf = RasPlanHdf.open_uri(plan_path)

    model_name = plan_path.split("/")[-1].split(".")[0]
    event_name = plan_hdf.get_plan_info_attrs()["Plan Name"]

    return model_name, event_name


def list_plan_hdfs(bucket: str, prefix: str) -> list:
    """List all plan HDF files in the given S3 prefix."""
    ras_files = list_keys_regex(s3_client=s3_client, bucket=bucket, prefix_includes=prefix)
    ras_files = [f"s3://{bucket}/{f}" for f in ras_files]
    plan_hdf_files = [f for f in ras_files if re.search(r"\.[p]\d{2}\.hdf$", f)]
    if plan_hdf_files:
        return plan_hdf_files
    else:
        raise ValueError(f"No plan hdf files found at bucket: {bucket} and prefix: {prefix} ")


def create_event_item(plan_file_path: str, source_model_path: str, output_prefix):

    model_name, event_name = extract_plan_info(plan_file_path)

    if "calibration" in event_name:
        logger.info(f"Creating stac item for event: {event_name}")

        short_event_name = event_name.split("_")[1]

        assets_prefix = f"{output_prefix}/model={model_name}/event={short_event_name}"
        dest_href = f"{output_prefix}/model={model_name}/event={short_event_name}/item.json"

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
        "plan_file_path": "s3://trinity-pilot/calibration/hydraulics/blw-elkhart/blw-elkhart.p03.hdf",
        "source_model_path": "s3://trinity-pilot/stac/prod-support/calibration/model=blw-elkhart/item.json",
        "output_prefix": "s3://trinity-pilot/stac/prod-support/calibration",
    }

    create_event_item(
        plan_file_path=config["plan_file_path"],
        source_model_path=config["source_model_path"],
        output_prefix=config["output_prefix"],
    )
