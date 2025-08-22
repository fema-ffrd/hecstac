"""Ras Calibration Check Script for S3 based FFRD models. Requires env variables for AWS credentials."""

import argparse
import json
import logging
import os

from dotenv import load_dotenv
from openpyxl.utils.exceptions import IllegalCharacterError

from hecstac.common.logger import get_logger, initialize_logger
from hecstac.common.utils import load_config
from hecstac.common.s3_utils import init_s3_resources
from hecstac.ras.ffrd_calibration_check import (
    RASModelCalibrationChecker,
    RASModelCalibrationError,
)


def parse_args():
    """
    Parse command-line arguments.

    Requires:
        --config: Path to a JSON configuration file or JSON string.

    Example JSON config:

        Single:
        '{"bucket": "trinity-pilot", "prefix": "calibration/hydraulics/bridgeport", "skip_hdf_files": false}'

        List:
        '[
            {"bucket": "trinity-pilot", "prefix": "calibration/hydraulics/bridgeport", "skip_hdf_files": true},
            {"bucket": "trinity-pilot", "prefix": "calibration/hydraulics/kickapoo", "ras_model_name": "trinity_1203_kickapoo", "skip_hdf_files": true}
        ]'

        Note: the ras_model_name **should** be the same as the last part of the prefix. Not enforcing this for FFRD currently to allow for production testing.
        See List example above and in example-configs.
    """
    parser = argparse.ArgumentParser(description="Create STAC Item(s) and Run QC for HEC-RAS model .prj file(s) on S3.")
    parser.add_argument("--config", required=True, help="JSON string or path to JSON config file.")
    return parser.parse_args()


def main(config: dict):
    """
    Create STAC Items and run QC.

    Args:
        config (dict): Configuration dictionary loaded from a JSON string.
    """
    try:
        _, s3_client, _ = init_s3_resources()

        calibration_checker = RASModelCalibrationChecker(
            s3_client=s3_client,
            bucket=config["bucket"],
            prefix=config["prefix"],
            ras_model_name=config.get("ras_model_name"),
            skip_hdf_files=config.get("skip_hdf_files", False),
            crs=config.get("crs"),
        )

        logger.info(f"Processing: {calibration_checker.ras_project_key}")

        try:
            ras_files = calibration_checker.parse_files()
        except FileNotFoundError as e:
            logger.error(str(e))
            raise RASModelCalibrationError(str(e))
        except (PermissionError, ConnectionError) as e:
            error_msg = f"S3 access issue for {calibration_checker.ras_project_key}: {e}"
            logger.error(error_msg)
            raise RASModelCalibrationError(error_msg)
        except Exception as e:
            error_msg = f"Error parsing RAS files for {calibration_checker.ras_project_key}: {e}"
            logger.error(error_msg)
            raise RASModelCalibrationError(error_msg)

        ras_item = calibration_checker.create_item(ras_files)
        calibration_checker.upload_metadata(ras_item)
        calibration_checker.run_qc(ras_item)
        logger.info(f"Completed: {calibration_checker.ras_project_key}")

    except RASModelCalibrationError as e:
        raise
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
    initialize_logger(json_logging=False, level=logging.INFO)
    logger = get_logger("main")

    try:
        dotenv_loaded = load_dotenv()
        if not dotenv_loaded:
            # TODO: Verify no issues occur on AWS if roles are in place and env vars are not set
            logger.warning(".env file not found or not loaded.")

        args = parse_args()

        try:
            configs = load_config(args.config)
        except ValueError as e:
            logger.error(f"Invalid config input: {e}")
            raise SystemExit(1)

        for idx, config in enumerate(configs, 1):
            try:
                main(config)
            except RASModelCalibrationError as e:
                continue
            except Exception as e:
                continue

    except Exception as e:
        logger.exception(f"Fatal error during initialization or processing: {e}")
        raise SystemExit(1)
