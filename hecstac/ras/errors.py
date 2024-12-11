class GeometryAssetInvalidCRSError(Exception):
    "Invalid crs provided to geometry asset"


class GeometryAssetMissingCRSError(Exception):
    "Required crs is missing from geometry asset definition"
