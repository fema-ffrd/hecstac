"""HEC-HMS STAC Item constants."""

GPD_WRITE_ENGINE = "fiona"  # Latest default as of 2024-11-11 seems to be "pyogrio" which is causing issues.
# 5 spaces, (key), colon, (val), ignoring whitespace before and after key and val, e.g. "     Version: 4.10"
ATTR_KEYVAL_GROUPER = r"^     (\S.*?)\s*:\s*(.*?)\s*$"
# 7 spaces, (key), colon, (val), ignoring whitespace before and after key and val, e.g. "          X Coordinate: 1893766.8845025832"
ATTR_NESTED_KEYVAL_GROUPER = r"^       (\S.*?)\s*:\s*(.*?)\s*$"

BC_LENGTH = 50  # length of the estimated BC lines. -for subbasin bc lines
BC_LINE_BUFFER = 100  # distance to move the BC lines up the subbasin connector lines.

NL_KEYS = [
    "Discretization",
    "Canopy",
    "Surface",
    "LossRate",
    "Transform",
    "Baseflow",
    "Route",
    "Flow Method",
    "Diverter",
    "Enable Sediment Routing",
    "Enable Quality Routing",
    "Begin Snow",
    "Base Temperature",
    "Melt Rate ATI-Cold Rate Table Name",
]
