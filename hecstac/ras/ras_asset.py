from typing import Callable

from pystac import Asset
from rashdf import RasGeomHdf, RasHdf, RasPlanHdf


class GenericAsset(Asset):
    @property
    def program_version(self) -> str:
        pass

    @property
    def short_summary(self) -> str:
        pass


class ProjectAsset(GenericAsset):
    @property
    def ras_title(self) -> str:
        pass

    @property
    def units(self) -> str:
        pass

    @property
    def plan_current(self) -> str:
        pass

    def _plan_files(self) -> list[str]:
        pass

    def _geometry_files(self) -> list[str]:
        pass

    def _steady_flow_files(self) -> list[str]:
        pass

    def _quasi_unsteady_flow_files(self) -> list[str]:
        pass

    def _unsteady_flow_files(self) -> list[str]:
        pass


class PlanAsset(GenericAsset):
    @property
    def primary_geometry(self) -> str:
        pass

    @property
    def primary_flow(self) -> str:
        pass

    @property
    def short_id(self) -> str:
        pass


class GeometryAsset(GenericAsset):
    @property
    def ras_title(self) -> str:
        pass

    @property
    def rivers(self) -> dict[str, "River"]:
        """A dictionary of river_name: River (class) for the rivers contained in the HEC-RAS geometry file."""
        pass

    @property
    def reaches(self) -> dict[str, "Reach"]:
        """A dictionary of the reaches contained in the HEC-RAS geometry file."""
        pass

    @property
    def junctions(self) -> dict[str, "Junction"]:
        """A dictionary of the junctions contained in the HEC-RAS geometry file."""
        pass

    @property
    def cross_sections(self) -> dict[str, "XS"]:
        """A dictionary of all the cross sections contained in the HEC-RAS geometry file."""
        pass

    @property
    def structures(self) -> dict[str, "Structure"]:
        """A dictionary of the structures contained in the HEC-RAS geometry file."""
        pass

    @property
    def storage_areas(self) -> dict[str, "StorageArea"]:
        """A dictionary of the storage areas contained in the HEC-RAS geometry file."""
        pass

    @property
    def connections(self) -> dict[str, "Connection"]:
        """A dictionary of the SA/2D connections contained in the HEC-RAS geometry file."""
        pass

    @property
    def datetimes(self):
        """Get the latest node last updated entry for this geometry"""
        pass


class SteadyFlowAsset(GenericAsset):
    @property
    def ras_title(self) -> str:
        pass

    @property
    def n_profiles(self) -> str:
        pass


class QuasiUnsteadyFlowAsset(GenericAsset):
    @property
    def ras_title(self) -> str:
        pass


class UnsteadyFlowAsset(GenericAsset):
    pass


class HdfAsset(Asset):
    # class to represent stac asset with properties shared between plan hdf and geom hdf
    def __init__(self, hdf_file: str, hdf_constructor: Callable[[str], RasHdf]):
        href = hdf_file
        title = ""
        description = ""
        media_type = ""
        roles = []
        extra_fields = {}
        self.hdf_object = hdf_constructor(hdf_file)
        self._root_attrs = None
        self._geom_attrs = None
        self._structures_attrs = None
        self._2d_flow_attrs = None
        super().__init__(href, title, description, media_type, roles, extra_fields)

    @property
    def version(self) -> str | None:
        # example property to show pattern: if attributes in which property is found is not loaded, load them
        # then use key for the property in the dictionary of attributes to retrieve the property
        if self._root_attrs == None:
            self._root_attrs = self.hdf_object.get_root_attrs()
        return self._root_attrs.get("version")


class PlanHdfAsset(HdfAsset):
    # class to represent stac asset for plan HDF file associated with model
    def __init__(self, hdf_file: str):
        super().__init__(hdf_file, RasPlanHdf)
        self.hdf_object: RasPlanHdf
        self._plan_info_attrs = None
        self._plan_parameters_attrs = None
        self._meteorology_attrs = None

    @property
    def plan_information_base_output_interval(self) -> str | None:
        # example property to show pattern: if attributes in which property is found is not loaded, load them
        # then use key for the property in the dictionary of attributes to retrieve the property
        if self._plan_info_attrs == None:
            self._plan_info_attrs = self.hdf_object.get_plan_info_attrs()
        return self._plan_info_attrs.get("base_output_interval")


class GeomHdfAsset(HdfAsset):
    # class to represent stac asset for geom HDF file associated with model
    def __init__(self, hdf_file: str):
        super().__init__(hdf_file, RasGeomHdf.open_uri)
        self.hdf_object: RasGeomHdf


# # I think that this should maybe not be an asset class and instead the item class should have a method which determines how the thumbnail is generated
# class ThumbnailAsset(Asset):
#     # class to represent stac asset for ras model thumbnail created for ras model stac item
#     def __init__(self, ):
#         href = ""
#         title = None
#         description = None
#         media_type = None
#         roles = None
#         extra_fields = None
#         super().__init__(href, title, description, media_type, roles, extra_fields)


class River:
    pass


class XS:
    pass


class Structure:
    pass


class Reach:
    pass


class Junction:
    pass


class StorageArea:
    pass


class Connection:
    pass
