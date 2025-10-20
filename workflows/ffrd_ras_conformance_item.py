from hecstac.common.s3_utils import init_s3_resources, list_keys_regex, save_bytes_s3, parse_s3_url
from hecstac.events.ffrd import FFRDEventItem
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

    Example JSON config:

        Single:
        '{
            "item_output_path": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork/item.json",
            "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/hydraulics/event_num=1/model=blw-clear-fork/flow.pq",
            "source_model_item": "s3://trinity-pilot/stac/prod-support/calibration/model=blw-clear-fork/item.json"
        }'

        List:
        '[
            {
            "item_output_path": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork/item.json",
            "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/hydraulics/event_num=1/model=blw-clear-fork/flow.pq",
            "source_model_item": "s3://trinity-pilot/stac/prod-support/calibration/model=blw-clear-fork/item.json"
            },
            {
            "item_output_path": "s3://trinity-pilot/conformance/simulations/event-data/150/hydraulics/bedias-creek/item.json",
            "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/hydraulics/event_num=150/model=bedias-creek/flow.pq",
            "source_model_item": "s3://trinity-pilot/stac/prod-support/calibration/model=bedias-creek/item.json"
            }
        ]'

    """
    parser = argparse.ArgumentParser(description="Create STAC Item(s) for HEC-RAS conformance model events on S3.")
    parser.add_argument("--config", required=True, help="JSON string or path to JSON config file.")
    return parser.parse_args()


def extract_model_and_event_id(path:str):
    """Extract model id and event id from path with parts 'event=' and 'model='. Zero-pads event id to 5 digits.
    
    example path = 's3://trinity-pilot/stac/hydraulics/event_num=1/model=blw-clear-fork/flow.pq'

    """
    event_id = None
    model_id = None

    for part in path.split('/'):
        if 'event_num=' in part:
            event_id = part.strip('event_num=')
        elif 'model=' in part:
            model_id = part.strip("model_id=")

    if event_id == None or model_id == None:
        raise ValueError(f"Could not extract model or event id from path: {path}. Path should contain 'event=' and 'model='. ")
    
    padded_event_id = event_id.zfill(5)

    return model_id, padded_event_id

def get_ras_files(s3_client, item_output_path):
    """Get files from a given s3 output path."""

    bucket, output_key = parse_s3_url(item_output_path)

    model_prefix = output_key.replace(output_key.split('/')[-1], '')
    ras_files = list_keys_regex(s3_client, bucket, model_prefix, return_full_path=True)

    file_to_remove = "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork/item.json"
    if file_to_remove in ras_files:
        ras_files.remove(file_to_remove)
        
    return ras_files


def create_conformance_item(item_output_path: str, ras_files: list, source_model_path: str, flow_output_path:str, item_id: str):
    """Create FFRD RAS conformance item and add flow parquet as an asset."""

    event_item = FFRDEventItem(ras_simulation_files = ras_files, source_model_paths=[source_model_path], event_id = item_id)
    event_item.add_flow_asset(flow_output_path)
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
        ras_files = get_ras_files(s3_client, config["item_output_path"])

        model_id, event_id = extract_model_and_event_id(config["flow_output_path"])

        item_id = f"{model_id}_e{event_id}"

        event_item = create_conformance_item(
            item_output_path=config['item_output_path'],
            ras_files=ras_files,
            source_model_path=config["source_model_item"],
            flow_output_path=config["flow_output_path"],
            item_id = item_id
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