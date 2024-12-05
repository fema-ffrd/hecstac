from ras_asset import (
    GenericAsset,
    GeometryAsset,
    PlanAsset,
    ProjectAsset,
    QuasiUnsteadyFlowAsset,
    RasGeomHdf,
    RasPlanHdf,
    SteadyFlowAsset,
    UnsteadyFlowAsset,
)


def asset_factory(
    filepath: str,
) -> (
    GenericAsset
    | GeometryAsset
    | PlanAsset
    | ProjectAsset
    | QuasiUnsteadyFlowAsset
    | RasGeomHdf
    | RasPlanHdf
    | SteadyFlowAsset
    | UnsteadyFlowAsset
):
    pass
