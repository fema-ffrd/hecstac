"""Errors for the ras module."""


class GeometryAssetInvalidCRSError(Exception):
    """Invalid crs provided to geometry asset."""


class GeometryAssetMissingCRSError(Exception):
    """Required crs is missing from geometry asset definition."""


class GeometryAssetNoXSError(Exception):
    """1D geometry asset has no cross sections; cross sections are required to calculate the goemetry of the asset."""
