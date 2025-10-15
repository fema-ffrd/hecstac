"""AWS S3 utlity functions."""

import os
from pathlib import Path

import boto3
import botocore
import fiona
from mypy_boto3_s3.service_resource import ObjectSummary
from pystac import Asset
from pystac.extensions.storage import StorageExtension


def file_location(file: str | Path) -> str:
    """Return the location of a file."""
    file = Path(file)
    if file.is_file() or file.is_dir():
        return "local"
    elif str(file).startswith("s3:"):
        return "s3"
    else:
        return "unknown"


def check_storage_extension(asset: Asset) -> Asset:
    """If the file is hosted on S3, add the storage extension."""
    if file_location(asset.href) == "s3":
        stor_ext = StorageExtension.ext(asset)
        meta = get_metadata(asset.href)
        stor_ext.apply(platform="AWS", region=meta["storage:region"], tier=meta["storage:tier"])
    return asset


def get_metadata(key: str) -> dict:
    """Read the head object and return metadata."""
    _, _, s3_resource = init_s3_resources2()
    bucket, key = split_s3_key(key)
    bucket = s3_resource.Bucket(bucket)
    key_obj = bucket.Object(key)
    return get_basic_object_metadata(key_obj)


def split_s3_key(s3_path: str) -> tuple[str, str]:
    """
    Split an S3 path into the bucket name and the key.

    Parameters
    ----------
        s3_path (str): The S3 path to split. It should be in the format 's3://bucket/key'.

    Returns
    -------
        tuple: A tuple containing the bucket name and the key. If the S3 path does not contain a key, the second element
          of the tuple will be None.

    The function performs the following steps:
        1. Removes the 's3://' prefix from the S3 path.
        2. Splits the remaining string on the first '/' character.
        3. Returns the first part as the bucket name and the second part as the key. If there is no '/', the key will be None.

    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"s3_path does not start with s3://: {s3_path}")
    bucket, _, key = s3_path[5:].partition("/")
    if not key:
        raise ValueError(f"s3_path contains bucket only, no key: {s3_path}")
    return bucket, key


def get_basic_object_metadata(obj: ObjectSummary) -> dict:
    """
    Retrieve basic metadata of an AWS S3 object.

    Parameters
    ----------
        obj (ObjectSummary): The AWS S3 object.

    Returns
    -------
        dict: A dictionary with the size, ETag, last modified date, storage platform, region, and
              storage tier of the object.
    """
    try:
        _ = obj.load()
        return {
            "file:size": obj.content_length,
            "e_tag": obj.e_tag.strip('"'),
            "last_modified": obj.last_modified.isoformat(),
            "storage:platform": "AWS",
            "storage:region": obj.meta.client.meta.region_name,
            "storage:tier": obj.storage_class,
        }
    except botocore.exceptions.ClientError:
        raise KeyError(f"Unable to access {obj.key} check that key exists and you have access")


def create_fiona_aws_session():
    """Create fiona s3 session."""
    return fiona.session.AWSSession(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        region_name=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")),
    )


def init_s3_resources2(minio_mode: bool = False):
    """Initialize s3 resources."""
    if minio_mode:
        session = boto3.Session(
            aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("MINIO_SECRET_ACCESS_KEY"),
        )

        s3_client = session.client("s3", endpoint_url=os.environ.get("MINIO_S3_ENDPOINT"))

        s3_resource = session.resource("s3", endpoint_url=os.environ.get("MINIO_S3_ENDPOINT"))

        return session, s3_client, s3_resource
    else:
        # Instantitate S3 resources
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        s3_client = session.client("s3")
        s3_resource = session.resource("s3")
        return session, s3_client, s3_resource
