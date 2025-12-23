"""Geometry utils."""

from pyproj import CRS, Transformer
from shapely import Geometry
from shapely.ops import transform


def reproject_geometry(geom: Geometry, src_crs: str, target_crs: str = "EPSG:4326") -> Geometry:
    """Convert geometry from source CRS to target CRS. Target CRS defaults to WGS84."""
    pyproj_src = CRS.from_user_input(src_crs)
    pyproj_target = CRS.from_user_input(target_crs)

    if pyproj_src != pyproj_target:
        transformer = Transformer.from_crs(pyproj_src, pyproj_target, always_xy=True)
        return transform(transformer.transform, geom)
    return geom


def read_crs_from_prj(prj_file: str) -> CRS:
    """Read CRS from a .prj file."""
    with open(prj_file, "r") as file:
        wkt = file.read()
    return CRS.from_wkt(wkt)
