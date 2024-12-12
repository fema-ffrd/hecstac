from urllib.parse import urlparse

from pyproj import CRS
from pystac.stac_io import DefaultStacIO

from .ras.ras_item import RasModelItem, ThumbnailParameter
from .utils.s3_utils import S3StacIO


def validate_crs(input_str: str) -> None:
    CRS.from_user_input(input_str)


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
    parser.add_argument("prj_file", type=str)
    parser.add_argument("output_uri", type=str)
    parser.add_argument("crs", type=str)
    parser.add_argument(
        "--thumbnail_source",
        type=str,
        choices=[p.value for p in ThumbnailParameter],
        default=ThumbnailParameter.XS.value,
    )

    args = parser.parse_args()

    validate_crs(args.crs)

    main(args.prj_file, args.output_uri, args.crs, ThumbnailParameter(args.thumbnail_source))
