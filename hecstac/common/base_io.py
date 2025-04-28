"""IO module to simplify s3 and local file handling."""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import obstore


class ModelFileReader:
    """A class to read model files from either the local file system or an S3 bucket."""

    def __init__(self, path: str | os.PathLike, store: Optional[obstore.store.ObjectStore] = None):
        """
        Initialize the ModelFileReader.

        Args:
            path : str | os.Pathlike
                The absolute path to the RAS file.
            store : obstore.store.ObjectStore, optional
                The obstore file system object. If not provided, it will use the S3 store.
        """
        if os.path.exists(path):
            self.local = True
            self.store = None
            self.path = Path(path)
            self.content = open(self.path, "r").read()
            # self.logger = get_logger(__name__)

        else:
            self.local = False
            parsed = urlparse(str(path))
            if parsed.scheme != "s3":
                raise ValueError(f"Expected S3 path, got: {path}")
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            self.store = store or obstore.store.S3Store(
                bucket=bucket,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            self.path = key
            self.content = (
                obstore.open_reader(self.store, self.path).readall().to_bytes().decode("utf-8").replace("\r\n", "\n")
            )
