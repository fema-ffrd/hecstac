"""Utility functions for the hecstac ras module."""

import logging
import os
from functools import wraps
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely import lib
from shapely.errors import UnsupportedGEOSVersionError
from shapely.geometry import LineString, MultiPoint, Point


def find_model_files(ras_prj: str) -> list[str]:
    """Find all files with same base name."""
    ras_prj = Path(ras_prj)
    parent = ras_prj.parent
    stem = Path(ras_prj).name.split(".")[0]
    return [str(i.as_posix()) for i in parent.glob(f"{stem}*")]


def is_ras_prj(url: str) -> bool:
    """Check if a file is a HEC-RAS project file."""
    with open(url) as f:
        file_str = f.read()
    if "Proj Title" in file_str.split("\n")[0]:
        return True
    else:
        return False


def search_contents(
    lines: list[str], search_string: str, token: str = "=", expect_one: bool = True, require_one: bool = True
) -> list[str] | str:
    """Split a line by a token and returns the second half of the line if the search_string is found in the first half."""
    results = []
    for line in lines:
        if f"{search_string}{token}" in line:
            results.append(line.split(token)[1])

    if expect_one and len(results) > 1:
        raise ValueError(f"expected 1 result for {search_string}, got {len(results)} results")
    elif require_one and len(results) == 0:
        raise ValueError(f"1 result for {search_string} is required, no results found")
    elif expect_one and len(results) == 1:
        return results[0]
    else:
        return results


def handle_spaces(line: str, lines: list[str]):
    """Handle spaces in the line."""
    if line in lines:
        return line
    elif handle_spaces_arround_equals(line.rstrip(" "), lines):
        return handle_spaces_arround_equals(line.rstrip(" "), lines)
    elif handle_spaces_arround_equals(line + " ", lines) in lines:
        return handle_spaces_arround_equals(line + " ", lines)
    else:
        raise ValueError(f"line: {line} not found in lines")


def handle_spaces_arround_equals(line: str, lines: list[str]) -> str:
    """Handle spaces in the line."""
    if line in lines:
        return line
    elif "= " in line:
        if line.replace("= ", "=") in lines:
            return line.replace("= ", "=")
    else:
        return line.replace("=", "= ")


def text_block_from_start_end_str(
    start_str: str, end_strs: list[str], lines: list, additional_lines: int = 0
) -> list[str]:
    """Search for an exact match to the start_str and return all lines from there to a line that contains the end_str."""
    start_str = handle_spaces(start_str, lines)

    start_index = lines.index(start_str)
    end_index = len(lines)
    for line in lines[start_index + 1 :]:
        if end_index != len(lines):
            break
        for end_str in end_strs:
            if end_str in line:
                end_index = lines.index(line) + additional_lines
                break
    return lines[start_index:end_index]


def text_block_from_start_str_to_empty_line(start_str: str, lines: list) -> list[str]:
    """Search for an exact match to the start_str and return all lines from there to the next empty line."""
    start_str = handle_spaces(start_str, lines)
    results = []
    in_block = False
    for line in lines:
        if line == start_str:
            in_block = True
            results.append(line)
            continue

        if in_block:
            if line == "":
                results.append(line)
                return results
            else:
                results.append(line)
    return results


def text_block_from_start_str_length(start_str: str, number_of_lines: int, lines: list[str]) -> list[str]:
    """Search for an exact match to the start token and return a number of lines equal to number_of_lines."""
    start_str = handle_spaces(start_str, lines)
    results = []
    in_block = False
    for line in lines:
        if line == start_str:
            in_block = True
            continue
        if in_block:
            if len(results) >= number_of_lines:
                return results
            else:
                results.append(line)


def data_pairs_from_text_block(lines: list[str], width: int) -> list[tuple[float, float]]:
    """Split lines at given width to get paired data string. Split the string in half and convert to tuple of floats."""
    pairs = []
    for line in lines:
        for i in range(0, len(line), width):
            x = line[i : int(i + width / 2)]
            y = line[int(i + width / 2) : int(i + width)]
            pairs.append((float(x), float(y)))

    return pairs


def delimited_pairs_to_lists(lines: list[str]) -> tuple[list[float], list[float]]:
    """Extract subdivisions from the manning's text block."""
    stations = []
    mannings = []
    for line in lines:
        pairs = line.split("       0")
        for p in pairs[:-1]:
            station = float(p[:8])
            n = float(p[8:])
            stations.append(station)
            mannings.append(n)
    return (stations, mannings)


def check_xs_direction(cross_sections: gpd.GeoDataFrame, reach: LineString):
    """Return only cross sections that are drawn right to left looking downstream."""
    river_reach_rs = []
    for _, xs in cross_sections.iterrows():
        try:
            point = reach.intersection(xs["geometry"])
            point = validate_point(point)
            xs_rs = reach.project(point)

            offset = xs.geometry.offset_curve(-1)
            if reach.intersects(offset):  # if the offset line intersects then use this logic
                point = reach.intersection(offset)
                point = validate_point(point)

                offset_rs = reach.project(point)
                if xs_rs > offset_rs:
                    river_reach_rs.append(xs["river_reach_rs"])
            else:  # if the original offset line did not intersect then try offsetting the other direction and applying
                # the opposite stationing logic; the orginal line may have gone beyound the other line.
                offset = xs.geometry.offset_curve(1)
                point = reach.intersection(offset)
                point = validate_point(point)

                offset_rs = reach.project(point)
                if xs_rs < offset_rs:
                    river_reach_rs.append(xs["river_reach_rs"])

        except IndexError as e:
            logging.debug(
                f"cross section does not intersect river-reach: {xs['river']} {xs['reach']} {xs['river_station']}: error: {e}"
            )
            continue
    return cross_sections.loc[cross_sections["river_reach_rs"].isin(river_reach_rs)]


def validate_point(geom):
    """Validate that point is of type Point. If Multipoint or Linestring create point from first coordinate pair."""
    if isinstance(geom, Point):
        return geom
    elif isinstance(geom, MultiPoint):
        return geom.geoms[0]
    elif isinstance(geom, LineString) and list(geom.coords):
        return Point(geom.coords[0])
    elif geom.is_empty:
        raise IndexError(f"expected point at xs-river intersection got: {type(geom)} | {geom}")
    else:
        raise TypeError(f"expected point at xs-river intersection got: {type(geom)} | {geom}")
