from collections import OrderedDict
from datetime import datetime

from hms.data_model import (
    Diversion,
    Junction,
    Reach,
    Reservoir,
    Sink,
    Source,
    Subbasin,
)

BC_LENGTH = 50  # length of the estimated BC lines. -for subbasin bc lines
BC_LINE_BUFFER = 100  # distance to move the BC lines up the subbasin connector lines.
GPD_WRITE_ENGINE = "fiona"  # Latest default as of 2024-11-11 seems to be "pyogrio" which is causing issues.
# 5 spaces, (key), colon, (val), ignoring whitespace before and after key and val, e.g. "     Version: 4.10"
ATTR_KEYVAL_GROUPER = r"^     (\S.*?)\s*:\s*(.*?)\s*$"
# 7 spaces, (key), colon, (val), ignoring whitespace before and after key and val, e.g. "          X Coordinate: 1893766.8845025832"
ATTR_NESTED_KEYVAL_GROUPER = r"^       (\S.*?)\s*:\s*(.*?)\s*$"


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

DEFAULT_BASIN_HEADER = {
    "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
    "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
    "Version": "4.11",
    "Filepath Separator": "\\",
    "Unit System": "English",
    "Missing Flow To Zero": "No",
    "Enable Flow Ratio": "No",
    "Compute Local Flow At Junctions": "No",
    "Unregulated Output Required": "No",
    "Enable Sediment Routing": "No",
}
DEFAULT_BASIN_LAYER_PROPERTIES = """     Element Layer:
          Name: Icons
          Layer shown: Yes
     End Layer:
     Element Layer:
          Name: Reaches
          Layer shown: Yes
     End Layer:
     2D Connection Layer:
          Layer shown: No
     End Layer:
     Element Layer:
          Name: Subbasins
          Layer shown: Yes
     End Layer:
     Discretization Layer:
          Layer shown: No
     End Layer:
     Breakpoint Layer:
          Layer shown: No
     End Layer:
     Base Layer:
          Name: Terrain
          Layer shown: No
          Properties Filename: maps\elevation4848081562023.xml
     End Layer:
"""
BASIN_DEFAULTS = {
    "Subbasin": {
        "attrs": {
            "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Latitude Degrees": "0",
            "Longitude Degrees": "0",
            "Canvas X": "",
            "Canvas Y": "",
            "Area": "",
            "Downstream": "",
            "Discretization": "Structured",
            "File": "",
            "Projection": "5070",
            "Cell Size": "2000.0",
            "Canopy": "Simple",
            "Allow Simultaneous Precip Et": "No",
            "Plant Uptake Method": "Simple",
            "Initial Canopy Storage Percent": "0",
            "Canopy Storage Capacity": "0",
            "Crop Coefficient": "1.0",
            "End Canopy": "",
            "Begin Snow": "Gridded Temperature Index",
            "Snow Water Equivalent Grid Name": "",
            "Cold Content Grid Name": "",
            "Liquid Water Grid Name": "",
            "Cold Content ATI Grid Name": "",
            "Melt Rate ATI Grid Name": "",
            "Base Temperature": "32",
            "Snow vs Rain Temperature": "38",
            "Dry Melt Rain Rate Limit": "0.5",
            "Cold Content ATI Snow Rate Limit": "0.2",
            "Cold Content ATI Decay Factor": "0.5",
            "Melt Rate ATI Decay Factor": "0.98",
            "Tree Line Elevation": "10000",
            "Interception Capacity": "0.0",
            "Liquid Water Holding Capacity": "4",
            "Ground Melt Method": "Fixed Value",
            "Ground Melt Rate": "0",
            "Dry Melt Rate Method": "ATI Function",
            "Wet Melt Method": "Fixed Value",
            "Rain Melt Rate": "0.1",
            "Melt Rate ATI-Cold Rate Table Name": "",
            "Melt Rate ATI-Melt Rate Table Name": "",
            "End Snow": "",
            "Surface": "None",
            "LossRate": "Deficit Constant",
            "Percent Impervious Area": "",
            "Initial Deficit": "",
            "Maximum Deficit": "",
            "Percolation Rate": "",
            "Recovery Factor": "1.0",
            "Transform": "Modified Clark",
            "Mod Clark Method": "Specified",
            "Time of Concentration": "",
            "Storage Coefficient": "",
            "Baseflow": "Linear Reservoir",
            "Groundwater Layer1": "1",
            "GW-1 Baseflow Fraction": "0.5",
            "GW-1 Number Reservoirs": "2",
            "GW-1 Routing Coefficient": "",
            "GW-1 Initial Flow/Area Ratio": "0",
            "Groundwater Layer2": "2",
            "GW-2 Baseflow Fraction": "0.5",
            "GW-2 Number Reservoirs": "3",
            "GW-2 Routing Coefficient": "",
            "GW-2 Initial Flow/Area Ratio": "0",
        },
        "gpkg_layer": "Subbasin",
        "sqlite_layer": "subbasin2d",
        "element": Subbasin,
    },
    "Junction": {
        "attrs": {
            "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Canvas X": "0",
            "Canvas Y": "0",
            "Downstream": "",
        },
        "gpkg_layer": "Junction",
        "sqlite_layer": "junction",
        "element": Junction,
    },
    "Reach": {
        "attrs": {
            "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Canvas X": "0",
            "Canvas Y": "0",
            "From Canvas X": "0",
            "From Canvas Y": "0",
            "Downstream": "",
            "Route": "Muskingum Cunge",
            "Channel": "8-point",
            "Length": "",
            "Energy Slope": "",
            "Mannings n": "",
            "Left Mannings n": "",
            "Right Mannings n": "",
            "Cross Section Name": "",
            "Initial Variable": "Combined Inflow",
            "Space-Time Method": "Automatic DX and DT",
            "Index Parameter Type": "Index Flow",
            "Index Flow": "500",
            "Number Subreaches": "1",
            "Maximum Depth Iterations": "20",
            "Maximum Route Step Iterations": "30",
            "Channel Loss": "None",
        },
        "gpkg_layer": "Reach",
        "sqlite_layer": "reach2d",
        "element": Reach,
    },
    "Sink": {
        "attrs": {
            "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Canvas X": "0",
            "Canvas Y": "0",
            "Computation Point": "No",
        },
        "gpkg_layer": "Sink",
        "sqlite_layer": "sink",
        "element": Sink,
    },
    "Reservoir": {
        "attrs": {
            "Description": "",
            "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Canvas X": "0",
            "Canvas Y": "0",
            "From Canvas X": "0",
            "From Canvas Y": "0",
            "Downstream": "",
            "Route": "Modified Puls",
            "Routing Curve": "",
            "Initial Storage": "0",
            "Storage-Outflow Table": "",
            "Elevation-Storage Table": "",
            "Primary Table": "",
        },
        "gpkg_layer": "Reservoir",
        "sqlite_layer": "reservoir",
        "element": Reservoir,
    },
    "Diversion": {"attrs": {}, "gpkg_layer": "Diversion", "sqlite_layer": "diversion", "element": Diversion},
    "Source": {
        "attrs": {
            "Last Modified Date": datetime.now().strftime("%#d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Canvas X": "0",
            "Canvas Y": "0",
            "From Canvas X": "0",
            "From Canvas Y": "0",
            "Downstream": "",
            "Flow Method": "",
            "Flow Gage": "",
            "End Flow Method": "",
        },
        "gpkg_layer": "Source",
        "sqlite_layer": "source",
        "element": Source,
    },
}

GRID_DEFAULTS = {
    "attrs": OrderedDict(
        {
            # "Grid": "",
            "Grid Type": "",
            "Last Modified Date": datetime.now().strftime("%d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Reference Height Units": "Meters",
            "Reference Height": "10.0",
            "Data Source Type": "External DSS",
            "Variant": "Variant-1",
            "  Last Variant Modified Date": datetime.now().strftime("%d %B %Y"),
            "  Last Variant Modified Time": datetime.now().strftime("%H:%M:%S"),
            "  Default Variant": "Yes",
            "  DSS File Name": "",
            "  DSS Pathname": "",
            "End Variant": "Variant-1",
            "Use Lookup Table": "No",
            # "End": "",
        }
    )
}

GAGE_DEFAULTS = {
    "attrs": OrderedDict(
        {
            # "Gage": "",
            "Last Modified Date": datetime.now().strftime("%d %B %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Reference Height Units": "Feet",
            "Reference Height": "32.808",
            "Gage Type": "",
            "Units": "CFS",
            "Data Type": "INST-VAL",
            "Data Source Type": "External DSS",
            "Variant": "Variant-1",
            "  Last Variant Modified Date": datetime.now().strftime("%d %B %Y"),
            "  Last Variant Modified Time": datetime.now().strftime("%H:%M:%S"),
            "  Default Variant": "Yes",
            "  DSS File Name": "",
            "  DSS Pathname": "",
            "  Start Time": "",
            "  End Time": "",
            "End Variant": "Variant-1",
            # "End": "",
        }
    )
}


CONTROL_DEFAULTS = {
    "attrs": OrderedDict(
        {
            # "Control": "",
            "Last Modified Date": datetime.now().strftime("%d %b %Y"),
            "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
            "Version": "4.11",
            "Time Id": "America/Denver",
            "Time Zone GMT Offset": "-25200000",
            "Start Date": "",
            "Start Time": "",
            "End Date": "",
            "End Time": "",
            "Time Interval": "15",
            # "End": "",
        }
    )
}

RUN_DEFAULTS = {
    "attrs": {
        # "Run": "",
        "Default Description": "Yes",
        "Log File": "",
        "DSS File": "",
        "Is Save Spatial Results": "No",
        "Last Modified Date": datetime.now().strftime("%d %b %Y"),
        "Last Modified Time": datetime.now().strftime("%H:%M:%S"),
        "Last Execution Date": "",
        "Last Execution Time": "",
        "Basin": "",
        "Precip": "",
        "Control": "",
        "Save State Type": "None",
        "Time-Series Output": "Save All",
        "Time Series Results Manager Start": "",
        "Time Series Results Manager End": "",
        # "End": "",
    }
}
