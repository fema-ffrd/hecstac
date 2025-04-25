import logging
import os
from pathlib import Path
from typing import Optional
import obstore
import fsspec
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ModelFileReader:
    ...
    """A class to read model files from either the local file system or an S3 bucket."""

    def __init__(self, path: str | os.PathLike, store: Optional[obstore.store.ObjectStore] = None):
        """
        Initializes the ModelFileReader.

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

        else:
            self.local = False
            parsed = urlparse(str(path))
            if parsed.scheme != "s3":
                raise ValueError(f"Expected S3 path, got: {path}")
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            self.store = store or obstore.store.S3Store(bucket=bucket)
            self.path = key
            self.content = (
                obstore.open_reader(self.store, self.path).readall().to_bytes().decode("utf-8").replace("\r\n", "\n")
            )


# def _get_fsspec_protocol(fs: fsspec.AbstractFileSystem) -> str:
#     """Get the protocol of the fsspec file system."""
#     if isinstance(fs.protocol, (list, tuple)):
#         return fs.protocol[0]
#     return fs.protocol


# class ModelFileReader:
#     """HEC-RAS model file class.

#     Represents a single file in a HEC-RAS model (project, geometry, plan, or flow file).

#     Attributes
#     ----------
#     path: Path to the file.
#     hdf_path: Path to the associated HDF file, if applicable.
#     """

#     fs: fsspec.AbstractFileSystem

#     def __init__(self, path: str | os.PathLike, fs: Optional[fsspec.AbstractFileSystem] = None):
#         """Instantiate a RasModelFile object by the file path.

#         Parameters
#         ----------
#         path : str | os.Pathlike
#             The absolute path to the RAS file.
#         fs : fsspec.AbstractFileSystem, optional
#             The fsspec file system object. If not provided, it will be created based on the path.
#         """
#         if fs:
#             self.fs = fs
#             self.path = Path(path)
#         else:
#             self.fs, _, fs_paths = fsspec.get_fs_token_paths(str(path))
#             self.path = fs_paths[0]
#         protocol = _get_fsspec_protocol(self.fs)
#         self.fsspec_path = f"{protocol}://{self.path}"

#     def contents(self):
#         """."""
#         with self.fs.open(self.fsspec_path, "r") as f:
#             return f.read()
