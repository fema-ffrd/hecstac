import logging
import os
from urllib.parse import urlparse

from pyproj import CRS
from pyproj.exceptions import CRSError
from pystac.stac_io import DefaultStacIO

from .ras.ras_item import RasModelItem, ThumbnailParameter
from .utils.s3_utils import S3StacIO


def validate_crs(crs_str: str) -> str:
    try:
        CRS.from_user_input(crs_str)
        return crs_str
    except CRSError:
        if os.path.exists(crs_str):
            with open(crs_str, "r") as f:
                crs = CRS.from_user_input(f.read())
                return crs.to_wkt()


def is_s3_uri(href: str) -> bool:
    parsed = urlparse(href)
    return parsed.scheme == "s3"


def main(prj_file: str, href: str, crs: str, thumbnail_parameter: ThumbnailParameter) -> None:
    if is_s3_uri(href):
        stac_io = S3StacIO()
    else:
        stac_io = DefaultStacIO()
    model_item = RasModelItem(prj_file, crs, href)
    model_item.populate()
    # model_item.validate()
    model_item.thumbnail(True, True, thumbnail_parameter)
    model_item.save_object(stac_io=stac_io)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prj_file", type=str, help="Project file to parse")
    parser.add_argument("output_uri", type=str, help="Location to store STAC item created as JSON document")
    parser.add_argument(
        "--crs",
        type=str,
        required=True,
        help="CRS authority code or path to file containing a proj4 or wkt of the CRS to use when parsing the project geometries",
    )
    parser.add_argument(
        "--thumbnail_source",
        type=str,
        choices=[p.value for p in ThumbnailParameter],
        default=ThumbnailParameter.XS.value,
    )

    args = parser.parse_args()

    logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.INFO)

    crs = validate_crs(args.crs)

    main(args.prj_file, args.output_uri, crs, ThumbnailParameter(args.thumbnail_source))
