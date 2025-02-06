"""Schema parsing."""

# TODO: update these, add imports, etc.
# def extract_schema_definition(definition_name: str) -> dict[str, Any]:
#     """Extract asset specific schema from ras extension schema"""

#     definitions: dict = deepcopy(RAS_EXTENSION_DICT["definitions"])
#     ras_schema = definitions[definition_name]
#     schema_specific_definitions = {}
#     for internal_definition_link in collect_definition_links(ras_schema):
#         internal_definition_name = os.path.basename(internal_definition_link)
#         definition_value = definitions[internal_definition_name]
#         schema_specific_definitions[internal_definition_name] = definition_value
#     if len(schema_specific_definitions) > 0:
#         ras_schema["definitions"] = schema_specific_definitions
#     with open(f"{definition_name}_schema.json", "w") as f:
#         json.dump(ras_schema, f)
#     return ras_schema


# def collect_definition_links(schema: dict[str, Any]) -> Iterator[str]:
#     for k, v in schema.items():
#         if k == "$ref":
#             if "#/definitions/" in v:
#                 yield v
#             elif "#" in v:
#                 raise ValueError(f"internal link found in key value pair {k}: {v} which is not found in #/definitions")
#         elif isinstance(v, dict):
#             yield from collect_definition_links(v)
#         elif isinstance(v, list):
#             for list_entry in v:
#                 if isinstance(list_entry, dict):
#                     yield from collect_definition_links(list_entry)


# #
# RAS_EXTENSION_PATH = os.path.join(os.path.dirname(__file__), "extension/schema.json")
# with open(RAS_EXTENSION_PATH, "r") as f:
#     data = json.load(f)
# RAS_EXTENSION_DICT: dict[str, Any] = data
