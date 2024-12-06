from urllib.parse import urlparse

from pystac.stac_io import DefaultStacIO
from ras.ras_item import RasModelItem, ThumbnailParameter
from utils.s3_utils import S3StacIO


def is_s3_uri(href: str) -> bool:
    parsed = urlparse(href)
    return parsed.scheme == "s3"


def main(prj_file: str, href: str, thumbnail_parameter: ThumbnailParameter) -> None:
    if is_s3_uri(href):
        stac_io = S3StacIO()
    else:
        stac_io = DefaultStacIO()
    model_item = RasModelItem(href, prj_file)
    model_item.autofind_project_assets()
    model_item.validate()
    model_item.thumbnail(True, True, ThumbnailParameter)
    model_item.save_object(stac_io=stac_io)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prj_file", type=str)
    parser.add_argument("output_uri", type=str)
    parser.add_argument("thumbnail_source", type=str, choices=[p.value for p in ThumbnailParameter])

    args = parser.parse_args()

    main(args.prj_file, args.output_uri, ThumbnailParameter(args.thumbnail_source))