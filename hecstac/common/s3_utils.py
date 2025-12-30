"""Utilities for S3."""

from __future__ import annotations

import io
import json
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.config import Config

from hecstac.common.logger import get_logger
from hecstac.common.consts import S3_PREFIX

if TYPE_CHECKING:
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


def list_keys_regex(
    s3_client: boto3.Session.client,
    bucket: str,
    prefix_includes: str,
    suffix: str = "",
    recursive: bool = True,
    return_full_path: bool = False,
) -> list:
    """List all keys in an S3 bucket matching a given prefix pattern and suffix."""
    keys = []
    prefix = prefix_includes.split("*")[0]  # Use the static part of the prefix for listing
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        kwargs["Delimiter"] = "/"

    prefix_pattern = re.compile(prefix_includes.replace("*", ".*"))

    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if prefix_pattern.match(key) and key.endswith(suffix):
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]

    if return_full_path:
        full_path_keys = [f"{S3_PREFIX}{bucket}/{key}" for key in keys]
        return full_path_keys
    else:
        return keys


def save_bytes_s3(
    data: io.BytesIO,
    s3_path: str,
    content_type: str = "",
    expected_bucket_owner: str | None = None,
):
    """Upload BytesIO to S3."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    expected_bucket_owner = expected_bucket_owner or os.getenv("AWS_EXPECTED_BUCKET_OWNER")

    params = {
        "Bucket": bucket,
        "Key": key,
        "Body": data.getvalue(),
    }
    if content_type:
        params["ContentType"] = content_type
    if expected_bucket_owner:
        params["ExpectedBucketOwner"] = expected_bucket_owner

    s3.put_object(**params)


def save_file_s3(
    local_path: str,
    s3_path: str,
):
    """Upload BytesIO to S3."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    expected_bucket_owner = expected_bucket_owner or os.getenv("AWS_EXPECTED_BUCKET_OWNER")

    extra_args = {}
    if expected_bucket_owner:
        extra_args["ExpectedBucketOwner"] = expected_bucket_owner

    s3.upload_file(
        Filename=local_path,
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra_args or None,
    )


def verify_file_exists(bucket: str, key: str, s3_client: boto3.client) -> bool:
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except Exception:
        raise FileNotFoundError(
            f"Cannot access file at `{S3_PREFIX}{bucket}/{key}` please check the path and ensure credentials are correct."
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
    expected_href = f"{S3_PREFIX}{bucket}/{prefix}/{metadata_part}/{model_name}.json"
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
    save_bytes_s3(buffer, s3_key, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def parse_s3_url(s3_url: str):
    """
    Extract the bucket name and path from an S3 URL.

    Args:
        s3_url (str): The S3 URL (e.g., 's3://my-bucket/path/to/object.txt').

    Returns
    -------
        tuple: (bucket_name, path)
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")
    return bucket, path


def make_uri_public(uri: str) -> str:
    """Convert from an AWS S3 URI to an https url."""
    bucket, path = parse_s3_url(uri)
    return f"https://{bucket}.s3.amazonaws.com/{path}"
