from dataclasses import dataclass


@dataclass
class Field:
    field_name: str
    type: str
    description: str
    required: bool


# Class for generating Markdown documents documenting extensions in such a way that they fall in line with template markdown file (https://github.com/stac-extensions/template/blob/main/README.md) with alterations where necessary or to reduce required manual input
# intention is to have properties which pull out headings and subheadings from schema
class ExtensionSchema:
    def __init__(self, schema_url: str) -> None:
        # reads schema url
        pass

    @property
    def title(self) -> str:
        pass

    @property
    def identifier(self) -> str:
        pass

    @property
    def field_name_prefix(self) -> str:
        pass

    @property
    def item_fields(self) -> list[Field]:
        pass

    @property
    def asset_definitions(self) -> dict[str, list[Field]]:
        # dict will have keys of patterns used to determine filetype of asset and values of the list of 'field' properties that are expected in files of that type
        pass

    @property
    def linked_definitions(self) -> dict[str, list[Field]]:
        # this will include all of the objects defined with titles and descriptions and whatnot that are referenced in either item fields or asset definitions
        pass

    def to_markdown(self, path: str) -> None:
        # parses schema to markdown and saves to path
        pass
