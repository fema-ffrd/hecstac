from pystac import MediaType

from hecstac.common.asset_factory import GenericAsset


class RunFileAsset(GenericAsset):
    """RunFile asset."""

    def __init__(self, href: str, *args, **kwargs):
        roles = ["run-file", "ras-file", MediaType.TEXT]
        description = "Run file for steady flow analysis which contains all the necessary input data required for the RAS computational engine."
        super().__init__(href, roles=roles, description=description, *args, **kwargs)
        self.regex_parse_str = r".+\.hyd\d{2}$"
