from utils.generate_schema_markdown import (
    ExtensionSchema,
    ExtensionSchemaAssetSpecific,
    FieldUsability,
)

from hecstac.ras.consts import SCHEMA_URI as RAS_SCHEMA_URI


def ras_schema_to_markdown(path: str) -> None:
    ras_usability = FieldUsability(False, True, True, True, False, False)
    extension_schema = ExtensionSchemaAssetSpecific(RAS_SCHEMA_URI, ras_usability)
    extension_schema.to_markdown(path)


def main(extension: str, path: str) -> None:
    match extension:
        case "ras":
            return ras_schema_to_markdown(path)
    raise ValueError(f"Unexpected value for extension: {extension}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("extension", type=str, choices=["ras"])
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    main(args.extension, args.path)
