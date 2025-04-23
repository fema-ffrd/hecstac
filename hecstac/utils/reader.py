import logging
import os
from pathlib import Path
from typing import Optional

import fsspec

logger = logging.getLogger(__name__)


def _get_fsspec_protocol(fs: fsspec.AbstractFileSystem) -> str:
    """Get the protocol of the fsspec file system."""
    if isinstance(fs.protocol, (list, tuple)):
        return fs.protocol[0]
    return fs.protocol


class ModelFileReader:
    """HEC-RAS model file class.

    Represents a single file in a HEC-RAS model (project, geometry, plan, or flow file).

    Attributes
    ----------
    path: Path to the file.
    hdf_path: Path to the associated HDF file, if applicable.
    """

    fs: fsspec.AbstractFileSystem

    def __init__(self, path: str | os.PathLike, fs: Optional[fsspec.AbstractFileSystem] = None):
        """Instantiate a RasModelFile object by the file path.

        Parameters
        ----------
        path : str | os.Pathlike
            The absolute path to the RAS file.
        fs : fsspec.AbstractFileSystem, optional
            The fsspec file system object. If not provided, it will be created based on the path.
        """
        if fs:
            self.fs = fs
            self.path = Path(path)
        else:
            self.fs, _, fs_paths = fsspec.get_fs_token_paths(str(path))
            self.path = fs_paths[0]
        protocol = _get_fsspec_protocol(self.fs)
        self.fsspec_path = f"{protocol}://{self.path}"

    def contents(self):
        """."""
        with self.fs.open(self.fsspec_path, "r") as f:
            return f.read()
