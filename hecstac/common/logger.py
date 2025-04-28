"""Logging utility and setup."""

import json
import logging
import sys
from re import L

SUPPRESS_LOGS = [
    "boto3",
    "botocore",
    "geopandas",
    "fiona",
    "rasterio",
    "pyogrio",
    "xarray",
    "shapely",
    "matplotlib",
    "aiobotocore",
    "fsspec",
]


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter."""

    def format(self, record):
        """Add name and format structure."""
        logger_name = record.name
        if logger_name == "root":
            logger_name = "hecstac"

        log_record = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": logger_name,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def initialize_logger(json_logging: bool = False, level: int = logging.INFO) -> logging.Logger:
    """Initialize and return the hecstac logger."""
    logger = logging.getLogger("hecstac")
    logger.setLevel(level)

    if not logger.handlers:
        if json_logging:

            class FlushStreamHandler(logging.StreamHandler):
                def emit(self, record):
                    super().emit(record)
                    self.flush()

            handler = FlushStreamHandler(sys.stdout)
            formatter = JsonFormatter()
            handler.setFormatter(formatter)
        else:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            handler.setFormatter(formatter)

        handler.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False  # don't double log to root

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        if not root_logger.handlers:
            root_logger.addHandler(handler)

    for package in SUPPRESS_LOGS:
        logging.getLogger(package).setLevel(logging.WARNING)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get hecstac logger or a child logger."""
    base_logger = logging.getLogger("hecstac")
    if name:
        return base_logger.getChild(name)
    return base_logger
