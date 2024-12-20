import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

import jsonschema
import requests

# Define schema which defines expected structure for extensions schemas
META_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "required": ["$schema", "$id", "title", "description", "oneOf", "definitions"],
    "properties": {
        "$schema": {"type": "string"},
        "$id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "oneOf": {"type": "array", "items": {"type": "object"}},
        "definitions": {"type": "object", "required": ["stac_extensions", "require_any_field", "fields"]},
    },
}

ASSET_SPECIFIC_META_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "required": ["$schema", "$id", "title", "description", "oneOf", "definitions"],
    "properties": {
        "$schema": {"type": "string"},
        "$id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "oneOf": {"type": "array", "items": {"type": "object"}},
        "definitions": {"type": "object", "required": ["stac_extensions", "require_any_field", "fields", "assets"]},
    },
}


@dataclass
class Field:
    field_name: str
    type: str | list[str]
    description: str
    required: bool | None

    def __post_init__(self) -> None:
        if self.required:
            self.table_description = f"**REQUIRED** {self.description}"
        else:
            self.table_description = self.description
        if isinstance(self.type, list):
            self.type_str = " \| ".join(self.type)
        else:
            self.type_str = self.type


@dataclass
class FieldUsability:
    catalog: bool
    collection_properties: bool
    collection_item_assets: bool
    item_properties: bool
    item_assets: bool
    links: bool

    def __post_init__(self):
        self.catalog_str = " "
        if self.catalog:
            self.catalog_str = "x"
        self.collection_properties_str = " "
        if self.collection_properties:
            self.collection_properties_str = "x"
        self.collection_item_assets_str = " "
        if self.collection_item_assets:
            self.collection_item_assets_str = "x"
        self.item_properties_str = " "
        if self.item_properties:
            self.item_properties_str = "x"
        self.item_assets_str = " "
        if self.item_assets:
            self.item_assets_str = "x"
        self.links_str = " "
        if self.links:
            self.links_str = "x"


# Class for generating Markdown documents documenting extensions in such a way that they fall in line with template markdown file (https://github.com/stac-extensions/template/blob/main/README.md) with alterations where necessary or to reduce required manual input
# intention is to have properties which pull out headings and subheadings from schema
class ExtensionSchema:
    def __init__(self, schema_url: str, field_usability: FieldUsability, prefix: str | None = None) -> None:
        # reads schema url
        self.identifier = schema_url
        self.field_usability = field_usability
        self._schema_str = read_schema_text_from_url(schema_url)
        self.schema = json.loads(self._schema_str)
        self.validate_schema()
        self._prefix = prefix
        self._not_common_definition_names: list[str] = ["fields", "stac_extensions", "require_any_field"]
        self._required_item_properties: list[str] | None = None

    def validate_schema(self) -> None:
        jsonschema.validate(self.schema, META_SCHEMA)

    @property
    def title(self) -> str:
        return self.schema["title"]

    @property
    def prefix(self) -> str:
        if self._prefix == None:
            prefix_pattern = re.compile(r"^\^\(\?\!(.+):\)$")
            prefix_exclusivity_pattern = list(self.schema["definitions"]["fields"]["patternProperties"].keys())[0]
            match = prefix_pattern.match(prefix_exclusivity_pattern)
            self._prefix = match.group(1)
        return self._prefix

    @property
    def item_property_definitions(self) -> Iterator[Field]:
        definition_schema = {"type": "object", "required": ["type", "description"]}
        for definition_key, definition_value in self.schema["definitions"]["fields"]["properties"].items():
            jsonschema.validate(definition_value, definition_schema, jsonschema.Draft7Validator)
            field = Field(
                definition_key,
                definition_value["type"],
                definition_value["description"],
                definition_key in self.required_item_properties,
            )
            yield field

    @property
    def common_definitions(self) -> Iterator[Field]:
        definition_schema = {"type": "object", "required": ["type", "description"]}
        for definition_key, definition_value in self.schema["definitions"].items():
            if definition_key not in self._not_common_definition_names:
                jsonschema.validate(definition_value, definition_schema, jsonschema.Draft7Validator)
                field = Field(definition_key, definition_value["type"], definition_value["description"], None)
                yield field

    def get_item_meta_schema(self) -> dict[str, Any]:
        for stac_type_subschema in self.schema["oneOf"]:
            if "type" not in stac_type_subschema:
                for subschema in stac_type_subschema["allOf"]:
                    if "properties" in subschema:
                        meta_properties = subschema["properties"]
                        type_definition = meta_properties["type"]
                        if type_definition == {"const": "Feature"}:
                            return meta_properties
        raise ValueError("Item meta schema was not found in expected location")

    @property
    def required_item_properties(self) -> list[str]:
        if self._required_item_properties == None:
            self.validate_item_metadata_schema()
        return self._required_item_properties

    def to_markdown(self, path: str | None) -> str:
        # parses schema to markdown and saves to path
        markdown_str = ""
        # write title
        markdown_str += f"# {self.title} Extension\n\n"
        # write overall metadata (title, identifier, prefix)
        markdown_str += f"- **Title:** {self.title}\n"
        markdown_str += f"- **Identifier:** {self.identifier}\n"
        markdown_str += f"- **Field Name Prefix:** {self.prefix}\n\n"
        # write short description for extension
        markdown_str += f"The {self.title} Extension is an extension to the [SpatioTemporal Asset Catalog](https://github.com/radiantearth/stac-spec) (STAC) specification. The purpose of the extension is to introduce vocabulary useful in describing RAS models as STAC items and assets.\n\n"
        # summarize field usability
        markdown_str += f"## Fields\n\nThe fields in the table below can be used in these parts of STAC documents:\n\n"
        markdown_str += f"- [{self.field_usability.catalog_str}] Catalogs\n"
        markdown_str += f"- [{self.field_usability.collection_properties_str}] Collection Properties\n"
        markdown_str += f"- [{self.field_usability.collection_item_assets_str}] Collection Item Assets\n"
        markdown_str += f"- [{self.field_usability.item_properties_str}] Item Properties\n"
        markdown_str += f"- [{self.field_usability.item_assets_str}] Item Assets\n"
        markdown_str += f"- [{self.field_usability.links_str}] Links\n\n"
        # overview of fields
        fields_table = self._fields_to_table(self.item_property_definitions)
        markdown_str += fields_table
        markdown_str += "\n\n### Additional Field Information\n\n"
        # common definitions
        common_table = self._fields_to_table(self.common_definitions)
        markdown_str += common_table
        if path:
            with open(path, "w") as f:
                f.write(markdown_str)
        return markdown_str

    @staticmethod
    def _fields_to_table(fields: list[Field]) -> str:
        max_field_name_length = len("Field Name")
        max_type_length = len("Type")
        max_description_length = len("Description")
        for field in fields:
            if len(field.field_name) > max_field_name_length:
                max_field_name_length = len(field.field_name)
            if len(field.type_str) > max_type_length:
                max_type_length = len(field.type_str)
            if len(field.table_description) > max_description_length:
                max_description_length = len(field.table_description)
        field_name_header = "Field Name".ljust(max_field_name_length)
        type_header = "Type".ljust(max_type_length)
        description_header = "Description".ljust(max_description_length)
        table = f"| {field_name_header} | {type_header} | {description_header} |"
        table += f"\n| {'-' * len(field_name_header)} | {'-' * len(type_header)} | {'-' * len(description_header)} |"
        for field in fields:
            row = f"\n| {field.field_name.ljust(max_field_name_length)} | {field.type_str.ljust(max_type_length)} | {field.table_description.ljust(max_description_length)} |"
            table += row
        return table

    def validate_item_metadata_schema(self) -> None:
        # validates that the inner schema used to validate the structure of a stac item is structured as expected (has required properties and a ref to definitions/fields or alternatively just a ref to definitions/fields if no properties are required)
        # also populates _required_item_properties
        definitions_fields_referenced = False
        required_property_names = None
        meta_properties = self.get_item_meta_schema()
        item_properties_schema = meta_properties["properties"]
        if "allOf" in item_properties_schema:
            subschemas = item_properties_schema["allOf"]
            if len(subschemas) != 2:
                raise ValueError(f"Expected 2 subschemas in item properties meta schema, got {len(subschemas)}")
            for subschema in item_properties_schema["allOf"]:
                if "$ref" in subschema:
                    reference = subschema["$ref"]
                    if reference != "#/definitions/fields":
                        raise ValueError(
                            f"Expected definitions/fields to hold all definitions for properties in item, got {reference}"
                        )
                    definitions_fields_referenced = True
                elif "required" in subschema:
                    required_property_names = subschema["required"]
                else:
                    raise ValueError(f"Subschema found with neither $ref nor required: {subschema}")
        else:
            required_property_names = []
            if "$ref" not in item_properties_schema:
                raise ValueError(
                    "Expected definitions/fields to hold all definitions for properties in item, instead $ref was not used"
                )
            reference = item_properties_schema["$ref"]
            if reference != "#/definitions/fields":
                raise ValueError(
                    f"Expected definitions/fields to hold all definitions for properties in item, got {reference}"
                )
            definitions_fields_referenced = True
        if definitions_fields_referenced == False:
            raise ValueError("Reference to definitions/fields never found")
        if required_property_names == None:
            raise TypeError("Required property names was neither set to an empty list nor pulled from schema")
        self._required_item_properties = required_property_names


class ExtensionSchemaAssetSpecific(ExtensionSchema):
    def __init__(self, schema_url, field_usability: FieldUsability, prefix=None):
        super().__init__(schema_url, field_usability, prefix)
        self.pattern_definition_dict = self.get_pattern_definition_dict()
        self._not_common_definition_names.append("assets")
        self._not_common_definition_names.extend(self.pattern_definition_dict.values())

    def get_pattern_definition_dict(self) -> dict[str, str]:
        pattern_to_definition_name_dict = {}
        self.validate_asset_metadata_schema()
        asset_definitions = self.schema["definitions"]["assets"]
        for pattern, ref in asset_definitions["patternProperties"].items():
            pattern_to_definition_name_dict[pattern] = ref["$ref"].replace("#/definitions/", "", 1)
        return pattern_to_definition_name_dict

    def validate_asset_metadata_schema(self) -> None:
        meta_properties = self.get_item_meta_schema()
        asset_properties_schema = meta_properties["assets"]
        if "$ref" not in asset_properties_schema:
            raise ValueError(
                "Expected definitions/assets to hold all definitions for properties expected in assets, instead $ref was not used"
            )
        reference = asset_properties_schema["$ref"]
        if reference != "#/definitions/assets":
            raise ValueError(
                f"Expected definitions/assets to hold all definitions for properties expected in assets, got {reference}"
            )

    @property
    def asset_definitions(self) -> Iterator[tuple[str, list[Field]]]:
        no_ref_definition_schema = {"type": "object", "required": ["type", "description"]}
        with_ref_definition_schema = {
            "type": "object",
            "required": ["allOf"],
            "properties": {
                "allOf": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "object", "required": ["description"]},
                            {"type": "object", "required": ["$ref"]},
                        ]
                    },
                }
            },
        }
        for pattern, definition_name in self.pattern_definition_dict.items():
            field_list: list[Field] = []
            meta_definition = self.schema["definitions"][definition_name]
            required_properties: list[str] = meta_definition.get("required", [])
            for asset_property_definition_name, asset_property_definition_value in meta_definition[
                "properties"
            ].items():
                try:
                    jsonschema.validate(
                        asset_property_definition_value, no_ref_definition_schema, jsonschema.Draft7Validator
                    )
                    field = Field(
                        asset_property_definition_name,
                        asset_property_definition_value["type"],
                        asset_property_definition_value["description"],
                        asset_property_definition_name in required_properties,
                    )
                    field_list.append(field)
                except jsonschema.ValidationError:
                    jsonschema.validate(
                        asset_property_definition_value, with_ref_definition_schema, jsonschema.Draft7Validator
                    )
                    asset_property_definition_description = None
                    asset_property_definition_type = None
                    for all_of_property in asset_property_definition_value["allOf"]:
                        if "description" in all_of_property:
                            asset_property_definition_description = all_of_property["description"]
                        elif "$ref" in all_of_property:
                            asset_property_definition_type = all_of_property["$ref"]
                    field = Field(
                        asset_property_definition_name,
                        asset_property_definition_type,
                        asset_property_definition_description,
                        asset_property_definition_name in required_properties,
                    )
                    field_list.append(field)
            yield pattern, field_list

    def to_markdown(self, path: str | None) -> str:
        markdown_str = super().to_markdown(None)
        markdown_str += "\n\n## Asset Properties with Pattern Matching\n\n"
        markdown_str += "This section describes the pattern used to match against the asset title along with the expected structure of that asset\n"
        for pattern, field_list in self.asset_definitions:
            markdown_str += f"\n### {pattern}\n\n"
            properties_table = self._fields_to_table(field_list)
            markdown_str += f"{properties_table}\n"
        if path:
            with open(path, "w") as f:
                f.write(markdown_str)
        return markdown_str


def read_schema_text_from_url(url: str) -> str:
    with requests.get(url) as resp:
        schema = resp.text
    return schema
