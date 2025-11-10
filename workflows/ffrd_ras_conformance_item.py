"""Creates STAC Item and adds flow time series asset representing an HEC-RAS FFRD conformance event."""

from hecstac.common.s3_utils import init_s3_resources, list_keys_regex, save_bytes_s3, parse_s3_url
from hecstac.events.ras_ffrd import FFRDEventItem
import io
import json
import argparse
from hecstac.common.logger import initialize_logger
from hecstac.common.utils import load_config
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Requires:
        --config: Path to a JSON configuration file or JSON string.

    Config fields:
        Required:
            - model_prefix: S3 path prefix to the RAS simulation files. Item will be written to this prefix.
            - timeseries_output_path: S3 path for the output flow time series parquet asset.
        Optional:
            - source_model_item: S3 path to a source model STAC Item the event is derived from
    Example JSON config:

        Single:
        '{
            "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork",
            "timeseries_output_path": "s3://trinity-pilot/stac/prod-support/conformance/hydraulics/event_num=1/model=blw-clear-fork/timeseries.pq",
            "source_model_item": "s3://trinity-pilot/stac/prod-support/calibration/model=blw-clear-fork/item.json"
        }'

        List:
        '[
            {
            "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork",
            "timeseries_output_path": "s3://trinity-pilot/stac/prod-support/conformance/hydraulics/event_num=1/model=blw-clear-fork/timeseries.pq",
            },
            {
            "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/150/hydraulics/bedias-creek",
            "timeseries_output_path": "s3://trinity-pilot/stac/prod-support/conformance/hydraulics/event_num=150/model=bedias-creek/timeseries.pq",
            }
        ]'

    """
    parser = argparse.ArgumentParser(description="Create STAC Item(s) for HEC-RAS conformance model events on S3.")
    parser.add_argument("--config", required=True, help="JSON string or path to JSON config file.")
    return parser.parse_args()


def extract_model_and_event_id(path: str):
    """Extract model id and event id from a model_prefix, padding the event id to 5 digits.

    Example:
      prefix = 's3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork'
      returns ('blw-clear-fork', '00001')
    """
    model_id = path.split("/")[-1]
    event_id = None

    for part in path.split("/"):
        if part.isdigit():
            event_id = part
            break

    if event_id is not None:
        padded_event_id = event_id.zfill(5)
    else:
        padded_event_id = None
        logger.warning(f"No event id found in flow output path: {path}")

    return model_id, padded_event_id


def get_ras_files(s3_client, model_prefix):
    """Get files from a given s3 output path."""
    bucket, prefix = parse_s3_url(model_prefix)

    ras_files = list_keys_regex(s3_client, bucket, prefix, return_full_path=True)
    ras_files = [f for f in ras_files if not f.lower().endswith(("item.json"))]
    return ras_files


def create_conformance_item(
    model_prefix: str, ras_files: list, timeseries_output_prefix: str, item_id: str, source_model_path: str = None
):
    """Create FFRD RAS conformance item and add flow parquet as an asset."""
    flow_output_path = f"{timeseries_output_prefix}/flow_timeseries.pq"
    stage_output_path = f"{timeseries_output_prefix}/stage_timeseries.pq"
    
    if source_model_path is not None:
        source_model_paths = [source_model_path]
    else:
        source_model_paths = None
    event_item = FFRDEventItem(ras_simulation_files=ras_files, event_id=item_id, source_model_paths=source_model_paths)
    event_item.add_flow_ts_asset(flow_output_path)
    event_item.add_stage_ts_asset(stage_output_path)

    item_output_path = f"{model_prefix}/item.json"
    event_item.set_self_href(item_output_path)

    return event_item


def main(config) -> None:
    """
    Create STAC Items and time series assets for RAS event model.

    Args:
        config (dict): Configuration dictionary loaded from a JSON string.
    """
    try:
        _, s3_client, _ = init_s3_resources()
        ras_files = get_ras_files(s3_client, config["model_prefix"])

        model_id, event_id = extract_model_and_event_id(config["model_prefix"])

        if event_id is not None:
            item_id = f"{model_id}_e{event_id}"
        else:
            item_id = model_id

        source_model_item = config.get("source_model_item")

        event_item = create_conformance_item(
            model_prefix=config["model_prefix"],
            ras_files=ras_files,
            timeseries_output_prefix=config["timeseries_output_prefix"],
            item_id=item_id,
            source_model_path=source_model_item,
        )
        item_dict = event_item.to_dict()
        item_bytes = io.BytesIO(json.dumps(item_dict, indent=2).encode("utf-8"))

        save_bytes_s3(data=item_bytes, s3_path=event_item.self_href, content_type="application/json")

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
