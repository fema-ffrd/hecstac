"""Utilities for S3."""

import io
import json
import os
import re
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.config import Config

from hecstac.common.logger import get_logger
from hecstac.ras.item import RASModelItem


def init_s3_resources() -> tuple:
    """Establish a boto3 session and return the session, S3 client, and S3 resource handles with optimized config."""
    boto_config = Config(
        retries={"max_attempts": 3, "mode": "standard"},  # Default is 10
        connect_timeout=3,  # Seconds to wait to establish connection
        read_timeout=10,  # Seconds to wait for a read
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )

    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    s3_client = session.client("s3", config=boto_config)
    s3_resource = session.resource("s3", config=boto_config)

    return session, s3_client, s3_resource


def list_keys_regex(s3_client: boto3.Session.client, bucket: str, prefix_includes: str, suffix="") -> list:
    """List all keys in an S3 bucket with a given prefix and suffix."""
    keys = []
    kwargs = {"Bucket": bucket, "Prefix": prefix_includes}
    prefix_pattern = re.compile(prefix_includes.replace("*", ".*"))
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        keys += [
            obj["Key"] for obj in resp["Contents"] if prefix_pattern.match(obj["Key"]) and obj["Key"].endswith(suffix)
        ]
        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break
    return keys


def save_bytes_s3(
    data: io.BytesIO,
    s3_path: str,
    content_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
):
    """Upload BytesIO to S3."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=data.getvalue(), ContentType=content_type)


def verify_file_exists(bucket: str, key: str, s3_client: boto3.client) -> bool:
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot access file at `s3://{bucket}/{key}` please check the path and ensure credentials are correct."
        )


def preload_assets(item: RASModelItem) -> RASModelItem:
    """Force preload of all assets to make to_dict() fast, and return item."""
    for asset in item.assets.values():
        _ = asset.extra_fields
        if getattr(asset, "__file_class__", None) is not None:
            try:
                _ = asset.file
            except Exception:
                pass
    return item


def metadata_to_s3(
    bucket: str,
    prefix: str,
    model_name: str,
    s3_client: boto3.client,
    item: RASModelItem,
    metadata_part: str = "metadata",
):
    """Upload the metadata JSON to S3."""
    logger = get_logger(__name__)
    expected_href = f"s3://{bucket}/{prefix}/{metadata_part}/{model_name}.json"
    if item.self_href != expected_href:
        raise ValueError(
            f"Item self href `{item.self_href}` does not match the provided S3 key `{expected_href}`. Please check the item."
        )
    else:
        item_dict = item.to_dict()
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/{metadata_part}/{model_name}.json",
            Body=json.dumps(item_dict, indent=2).encode("utf-8"),
            ContentType="application/json",
        )


def qc_results_to_excel_s3(results: dict, s3_key: str) -> None:
    """Create an Excel file from RasqcResults JSON. with 2 sheets: passed and failed."""

    def flatten(group_name):
        rows = []
        for pattern, files in results.get(group_name, {}).items():
            for file, props in files.items():
                for prop in props:
                    rows.append({"Pattern Name": pattern, "File Name": file, "RAS Property Name": prop})
        return pd.DataFrame(rows)

    passed_df = flatten("passed")
    failed_df = flatten("failed")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        failed_df.to_excel(writer, sheet_name="failed", index=False)
        passed_df.to_excel(writer, sheet_name="passed", index=False)

    buffer.seek(0)
    save_bytes_s3(buffer, s3_key)
