"""Logging utility and setup."""

import logging
import sys
from re import L

SUPPRESS_LOGS = ["boto3", "botocore", "geopandas", "fiona", "rasterio", "pyogrio", "xarray", "shapely", "matplotlib"]


def initialize_logger(json_logging: bool = False, level: int = logging.INFO):
    """Initialize the ras logger."""
    logger = logging.getLogger("hecstac")
    logger.setLevel(level)
    if json_logging:
        for module in SUPPRESS_LOGS:
            logging.getLogger(module).setLevel(logging.WARNING)

        class FlushStreamHandler(logging.StreamHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        handler = FlushStreamHandler(sys.stdout)

        handler.setLevel(level)

        datefmt = "%Y-%m-%dT%H:%M:%SZ"
        fmt = """{"time": "%(asctime)s" , "level": "%(levelname)s", "msg": "%(message)s"}"""
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
    else:
        for package in SUPPRESS_LOGS:
            logging.getLogger(package).setLevel(logging.ERROR)
