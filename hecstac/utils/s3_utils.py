from urllib.parse import urlparse
import os
import boto3
from mypy_boto3_s3 import S3ServiceResource
from pystac.stac_io import DefaultStacIO


class S3StacIO(DefaultStacIO):
    def __init__(self, headers=None):
        super().__init__(headers)
        self.session = boto3.Session()
        self.s3: S3ServiceResource = self.session.resource("s3")

    def read_text(self, source: str, *_, **__) -> str:
        parsed = urlparse(url=source)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path[1:]
            obj = self.s3.Object(bucket, key)
            data_encoded: bytes = obj.get()["Body"].read()
            data_decoded = data_encoded.decode()
            return data_decoded
        else:
            return super().read_text(source)

    def write_text(self, dest: str, txt, *_, **__) -> None:
        parsed = urlparse(url=dest)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path[1:]
            obj = self.s3.Object(bucket, key)
            obj.put(Body=txt, ContentEncoding="utf-8")
        else:
            return super().write_text(dest, txt, *_, **__)


def split_s3_path(s3_path: str) -> tuple[str, str]:
    """Split an S3 path into the bucket name and the key.

    Parameters
    ----------
        s3_path (str): The S3 path to split. It should be in the format 's3://bucket/key'.

    Returns
    -------
        tuple: A tuple containing the bucket name and the key. If the S3 path does not contain a key, the second element
          of the tuple will be None.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"s3_path does not start with s3://: {s3_path}")
    bucket, _, key = s3_path[5:].partition("/")
    if not key:
        raise ValueError(f"s3_path contains bucket only, no key: {s3_path}")
    return bucket, key


def init_s3_resources():
    """Create a Boto3 session using AWS credentials from environment variables and creates both an S3 client and S3 resource for interacting with AWS S3."""
    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    s3_client = session.client("s3")
    s3_resource = session.resource("s3")
    return session, s3_client, s3_resource


def save_bytes_s3(byte_obj: bytes, s3_key: str) -> None:
    """Save bytes to S3."""
    _, s3_client, _ = init_s3_resources()
    bucket, key = split_s3_path(s3_key)
    s3_client.put_object(Body=byte_obj, Bucket=bucket, Key=key)
