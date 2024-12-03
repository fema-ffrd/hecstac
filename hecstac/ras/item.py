from asset import (
    GenericAsset,
    GeometryAsset,
    PlanAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
)
from pystac import Item


class RasModelItem(Item):
    def __init__(self, prj_file: str) -> None:
        pass
