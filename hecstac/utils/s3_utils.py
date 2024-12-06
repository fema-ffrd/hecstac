from urllib.parse import urlparse

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
