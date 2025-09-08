"""HEC-HMS STAC Item utlity functions."""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

from shapely import MultiPolygon, Polygon

from hecstac.common.logger import get_logger
from hecstac.hms.consts import ATTR_KEYVAL_GROUPER, ATTR_NESTED_KEYVAL_GROUPER, NL_KEYS

logger = get_logger(__name__)


def add_no_duplicate(d: dict, key, val):
    """Insert a key into a dictionary, logging a warning if the key already exists."""
    if key in d:
        return
    d[key] = val


def get_lines_until_end_sentinel(lines: list[str]) -> list[str]:
    """Retrieve all lines until the End point."""
    lines_found = []
    for line in lines:
        if line.strip() in ["End:", "End Computation Point:"]:
            break
        lines_found.append(line)
    else:
        raise ValueError("never found End:")
    return lines_found


def handle_special_cases(key, val):
    """Handle special cases."""
    if key == "Groundwater Layer":
        key = key + val
    elif "Groundwater Layer" in key:
        key = key.rstrip(val)
    elif key == "Use Basin Model":
        key = key + val
    elif "Use Basin Model" in key:
        key = key.rstrip(val)
    return key, val


def parse_attrs(lines: list[str]) -> OrderedDict:
    """Scan the lines down to the first instance of 'End:' and return dict containing all of the colon-separated keyval pair."""
    attrs = {}
    for line in lines:
        if line.strip() == "End:":
            break
        if not line:
            continue

        keyval_pairs = re.findall(ATTR_KEYVAL_GROUPER, line)
        if keyval_pairs:
            key, val = _process_keyval_pairs(attrs, keyval_pairs)

        # test if nested (10 spaces to start the line). if so create nested dictionary
        nested_keyval_pairs = re.findall(ATTR_NESTED_KEYVAL_GROUPER, line)
        if nested_keyval_pairs:
            _process_nested_pair(attrs, nested_keyval_pairs, key, val)

        key, val = _process_hamon_coefficient(attrs, line)
        keyval_pairs = (key, val)

        _handle_keyval_error(line, keyval_pairs, nested_keyval_pairs)
    else:
        raise ValueError("never found End:")
    return OrderedDict(attrs)


def _process_keyval_pairs(attrs: dict, keyval_pairs: list[any]) -> tuple[str, str]:
    """Process a standard key: value pair."""
    if len(keyval_pairs) != 1:
        raise ValueError(f"Expected 0 or 1 pairs, got {len(keyval_pairs)}: {keyval_pairs}")
    key, val = keyval_pairs[0]
    key, val = handle_special_cases(key, val)
    add_no_duplicate(attrs, key, val)

    return (key, val)


def _process_nested_pair(attrs: dict, nested_keyval_pairs: list[any], parent_key: str, parent_val: str):
    """Process a nested key-value pair."""
    if len(nested_keyval_pairs) != 1:
        raise ValueError(f"Expected 0 or 1 pairs, got {len(nested_keyval_pairs)}: {nested_keyval_pairs}")
    nested_key, nested_val = nested_keyval_pairs[0]
    # if attrs[key]val is a string then this is the first item in nested dictionary; change to dictionary instead of string.
    if isinstance(attrs[parent_key], str):
        attrs[parent_key] = {parent_val: {}}
    add_no_duplicate(attrs[parent_key][parent_val], nested_key, nested_val)


def _process_hamon_coefficient(attrs: dict, line: str) -> tuple[str, str]:
    if "Hamon Coefficient" in line:
        key, val = line.split(":")
        add_no_duplicate(attrs, key, val)

    return (key, val)


def _handle_keyval_error(line, keyval_pairs, nested_keyval_pairs):
    if not (keyval_pairs or nested_keyval_pairs):
        raise ValueError(f"unexpected line (does not have keyval nor nested keyval pair): {repr(line)}")
    if keyval_pairs and nested_keyval_pairs:
        raise RuntimeError(f"regex matched top-level and nested keyval pairs (pattern is bugged): {repr(line)}")


def remove_holes(geom):
    """Remove holes in the geometry."""
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    elif isinstance(geom, MultiPolygon):
        areas, polygons = [], []
        for geom in geom.geoms:
            polygons.append(Polygon(geom.exterior))
            areas.append(polygons[-1].area)
        idx = areas.index(max(areas))
        return polygons[idx]
    elif geom is None:
        return geom
    else:
        raise TypeError(f"cannot handle geometry type {type(geom)}. only able to handle Polygon and Multipolygon")


def attrs2list(attrs: OrderedDict) -> list[str]:
    """Convert dictionary of attributes to a list."""
    content = []
    for key, val in attrs.items():
        if isinstance(val, str):
            continue

        if isinstance(val, int) or isinstance(val, float):
            val = str(val)
        elif isinstance(val, dict):
            content += _serialize_dict(key, val)
            continue
        else:
            raise TypeError(
                f"attribute is not of type string. we are currently unable to serialize nested dicts. key, value = {repr(key)}, {repr(val)}"
            )

        content += _format_value(key, val)

    return content


def _serialize_dict(key, val) -> list:
    content = []

    value = list(val.keys())[0]
    content += [f"     {key}: {value}"]
    for k, v in val[value].items():
        content += [f"       {k}: {v}"]

    return content


def _format_value(key, val) -> list:
    content = []

    if key in NL_KEYS:
        content += [""]
    if key in ["Latitude Degrees", "Longitude Degrees"] and len(val.split(".")[0]) == 2:
        val = " " + val

    key, val = handle_special_cases(key, val)
    content += [f"     {key}: {val}"]

    return content


def insert_after_key(dic: dict, insert_key: str, new_key: str, new_val: str) -> OrderedDict:
    """Recreate the dictionary to insert key-val after the occurance of the insert_key if key-val doesn't exist yet in the dictionary."""
    new_dic = {}
    for key, val in dic.items():
        if key == new_key:
            continue
        new_dic[key] = val
        if key == insert_key:
            new_dic[new_key] = new_val
    return OrderedDict(new_dic)


def search_contents(lines: list, search_string: str, token: str = "=", expect_one: bool = True) -> list[str]:
    """Split a line by a token and returns the second half of the line if the search_string is found in the first half."""
    results = []
    for line in lines:
        if f"{search_string}{token}" in line:
            results.append(line.split(token)[1])

    if expect_one and len(results) > 1:
        raise ValueError(f"expected 1 result, got {len(results)}")
    elif expect_one and len(results) == 0:
        raise ValueError("expected 1 result, no results found")
    elif expect_one and len(results) == 1:
        return results[0]
    else:
        return results


class StacPathManager:
    """Build consistent paths for STAC items and collections assuming a top level local catalog."""

    def __init__(self, local_catalog_dir: str):
        self._catalog_dir = local_catalog_dir

    @property
    def catalog_dir(self):
        """Return the catalog directory."""
        return self._catalog_dir

    @property
    def catalog_file(self):
        """Return the catalog file path."""
        return f"{self._catalog_dir}/catalog.json"

    def catalog_item(self, item_id: str) -> str:
        """Return the catalog item file path."""
        return f"{self.catalog_dir}/{item_id}/{item_id}.json"

    def catalog_asset(self, item_id: str, asset_dir: str = "hydro_domains") -> str:
        """Return the catalog asset file path."""
        return f"{self.catalog_dir}/{asset_dir}/{item_id}.json"

    def collection_file(self, collection_id: str) -> str:
        """Return the collection file path."""
        return f"{self.catalog_dir}/{collection_id}/collection.json"

    def collection_dir(self, collection_id: str) -> str:
        """Return the collection directory."""
        return f"{self.catalog_dir}/{collection_id}"

    def collection_asset(self, collection_id: str, filename: str) -> str:
        """Return the collection asset filepath."""
        return f"{self.catalog_dir}/{collection_id}/{filename}"

    def collection_item_dir(self, collection_id: str, item_id: str) -> str:
        """Return the collection item directory."""
        return f"{self.catalog_dir}/{collection_id}/{item_id}"

    def collection_item(self, collection_id: str, item_id: str) -> str:
        """Return the collection item filepath."""
        return f"{self.catalog_dir}/{collection_id}/{item_id}/{item_id}.json"

    def collection_item_asset(self, collection_id: str, item_id: str, filename: str) -> str:
        """Return the collection item asset filepath."""
        return f"{self.catalog_dir}/{collection_id}/{item_id}/{filename}"
