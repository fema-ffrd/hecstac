"""Features developed duirng R&D for consideration in the hecstac package."""

import re
from pathlib import Path

import boto3

from hecstac.common.logger import get_logger
from hecstac.common.s3_utils import (
    list_keys_regex,
    metadata_to_s3,
    qc_results_to_excel_s3,
    verify_file_exists,
)
from hecstac.ras.item import RASModelItem
from rasqc.check import check
from rasqc.utils import summarize_results


def build_s3_path(
    bucket: str,
    prefix: str,
    model_name: str = "",
    metadata_part: str = "metadata",
    suffix: str = "",
    extension: str = "",
) -> str:
    """Make consistent s3 paths for metadata."""
    path = f"s3://{bucket}/{prefix}/{metadata_part}"
    if model_name:
        path += f"/{model_name}"
    if suffix:
        path += f"_{suffix}"
    if extension:
        path += f".{extension.lstrip('.')}"
    return path


class RASModelCalibrationError(Exception):
    """Custom exception for calibration processing errors."""

    pass


class RASModelCalibrationChecker:
    """Utility for streamlining QC checks for FFRD."""

    def __init__(
        self,
        s3_client: boto3.Session.client,
        bucket: str,
        prefix: str,
        ras_model_name: str = None,
        skip_hdf_files: bool = False,
        crs=None,
    ):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix
        self.crs = crs
        self.skip_hdf_files = skip_hdf_files

        s3_parts = Path(prefix).parts
        self.ras_model_name = ras_model_name or s3_parts[-1]
        self.ras_project_key = f"{self.prefix}/{self.ras_model_name}.prj"
        self.item_id = Path(self.ras_project_key).stem
        self.logger = get_logger(__name__)

    def parse_files(self):
        """Parse s3 keys to identify RAS files."""
        verify_file_exists(bucket=self.bucket, key=self.ras_project_key, s3_client=self.s3_client)

        ras_prefix = f"{self.prefix}/{self.ras_model_name}"
        ras_files = list_keys_regex(s3_client=self.s3_client, bucket=self.bucket, prefix_includes=ras_prefix)
        ras_files = [f"s3://{self.bucket}/{f}" for f in ras_files]

        if self.skip_hdf_files:
            ras_files = [f for f in ras_files if not f.endswith(".hdf")]

        return ras_files

    def create_item(self, ras_files):
        """Create a STAC Item from a RAS model."""
        ras_item = RASModelItem.from_prj(
            build_s3_path(self.bucket, self.prefix, self.ras_model_name),
            crs=self.crs,
            assets=ras_files,
        )

        thumbnail_dst = build_s3_path(
            self.bucket,
            self.prefix,
        )

        ras_item.add_model_thumbnails(
            ["mesh_areas", "breaklines", "bc_lines"],
            thumbnail_dest=thumbnail_dst,
        )

        item_json = build_s3_path(
            self.bucket,
            self.prefix,
            self.ras_model_name,
            extension="json",
        )
        ras_item.set_self_href(item_json)
        ras_item.validate()

        return ras_item

    def upload_metadata(self, ras_item):
        """Upload metadata output."""
        self.logger.debug("upload_metadata starting")
        metadata_to_s3(
            bucket=self.bucket,
            prefix=self.prefix,
            model_name=self.ras_model_name,
            s3_client=self.s3_client,
            item=ras_item,
        )
        self.logger.info("upload_metadata complete")

    def run_qc(self, ras_item):
        """Perform QC check and upload to s3."""
        qc_results = summarize_results(check(ras_item, check_suite="ras_stac_ffrd"))
        self.logger.debug("run_qc results computed")
        qc_results_path = build_s3_path(
            self.bucket,
            self.prefix,
            self.ras_model_name,
            suffix="qc-results",
            extension="xlsx",
        )
        qc_results_to_excel_s3(qc_results, qc_results_path)
        self.logger.info("run_qc completed upload")
