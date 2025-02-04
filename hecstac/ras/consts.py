"""Constants."""

import datetime
from shapely import Polygon, to_geojson, box
import json


SCHEMA_URI = (
    "https://raw.githubusercontent.com/fema-ffrd/hecstac/refs/heads/port-ras-stac/hecstac/ras/extension/schema.json"
)

NULL_DATETIME = datetime.datetime(9999, 9, 9)
NULL_GEOMETRY = Polygon()
NULL_STAC_GEOMETRY = json.loads(to_geojson(NULL_GEOMETRY))
NULL_BBOX = box(0, 0, 0, 0)
NULL_STAC_BBOX = NULL_BBOX.bounds
PLACEHOLDER_ID = "id"
