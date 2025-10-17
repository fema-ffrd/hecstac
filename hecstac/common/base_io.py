"""IO module to simplify s3 and local file handling."""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import obstore
from dotenv import load_dotenv

from hecstac.common.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


class ModelFileReaderError(Exception):
    """Return Model read error."""

    pass


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
        self.logger = get_logger(__name__)

        if os.path.exists(path):
            self._init_local(path)
        else:
            self._init_s3(path, store)

    def _init_local(self, path: str | os.PathLike) -> None:
        self.local = True
        self.store = None
        self.path = Path(path)
        try:
            self.content = open(self.path, "r").read()
        except UnicodeDecodeError as e:
            logger.warning(f"File contains invalid utf-8 characters at byte {e.start}.")
            self.content = open(self.path, "r", errors="ignore").read()

    def _init_s3(self, path: str | os.PathLike, store: Optional[obstore.store.ObjectStore]) -> None:
        self.local = False
        parsed = urlparse(str(path))
        if parsed.scheme != "s3":
            raise ValueError(f"Expected S3 path, got: {path}")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        self.store = store or self._create_s3_store(bucket)
        self.path = key
        try:
            for i in ["utf-8", "latin_1", "iso8859_15"]:
                try:
                    self.content = (
                        obstore.open_reader(self.store, self.path).readall().to_bytes().decode(i).replace("\r\n", "\n")
                    )
                    break
                except UnicodeDecodeError as e:
                    error_msg = f"Error parsing {self.path} with {i}: {e}"
            else:
                raise ModelFileReaderError(error_msg)
        except Exception as e:
            raise ModelFileReaderError(f"An unexpected error occurred: {e}")

    def _create_s3_store(self, bucket: str) -> obstore.store.S3Store:
        return obstore.store.S3Store(
            bucket=bucket,
            skip_signature=(os.getenv("AWS_ACCESS_KEY_ID") is None and os.getenv("AWS_SECRET_ACCESS_KEY") is None),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or "",
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or "",
            session_token=os.getenv("AWS_SESSION_TOKEN") or "",
        )
