from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

import hms.utils as utils
from shapely.geometry import LineString, Point, Polygon


@dataclass
class Element:
    """Parent class of basin elements (Subbasins, Reaches, etc)"""

    name: str
    attrs: OrderedDict


@dataclass
class BasinHeader:
    """Header of .basin"""

    attrs: dict


@dataclass
class BasinLayerProperties:
    """Part of footer of .basin, find via 'Basin Layer Properties:'.
    Data is stored as a series of layers rather than a set of attributes, so just storing the raw content for now.
    """

    content: str


@dataclass
class Control(Element):
    pass


@dataclass
class Grid(Element):
    pass


@dataclass
class Precipitation(Element):
    pass


@dataclass
class Temperature(Element):
    pass


@dataclass
class ET(Element):
    pass


@dataclass
class Subbasin_ET(Element):
    pass


@dataclass
class Gage(Element):
    pass


@dataclass
class ComputationPoints:
    """Part of footer of .basin, find via 'Computation Points:'.
    Data has some complex attributes with nested end-flags, so just storing raw content for now.
    """

    content: str


@dataclass
class BasinSpatialProperties:
    """Part of footer of .basin, find via 'Basin Spatial Properties:'.
    Data has some complex attributes with nested end-flags, so just storing raw content for now.
    """

    content: str


@dataclass
class BasinSchematicProperties:
    """Part of footer of .basin, find via 'Basin Schematic Properties:'"""

    attrs: dict


@dataclass
class Run:
    """Runs contained in the .run file."""

    name: str
    attrs: dict


@dataclass
class Subbasin(Element):
    geom: Polygon = None


@dataclass
class Table(Element):
    pass


@dataclass
class Pattern(Element):
    pass


@dataclass
class Reach(Element):
    geom: LineString = None
    slope: float = (
        None  # assumed units of the coordinate system is the same as what is used for the project.. need to confirm this assumption
    )


@dataclass
class Junction(Element):
    geom: Point = None


@dataclass
class Sink(Element):
    geom: Point = None


@dataclass
class Reservoir(Element):
    geom: Point = None


@dataclass
class Source(Element):
    geom: Point = None


@dataclass
class Diversion(Element):
    geom: Point = None


class ElementSet:
    """Behaves like a dictionary of Basin elements (Subbasins, Reaches, etc) with key conflict checking."""

    def __init__(self):
        self.elements: dict[str, Element] = {}
        self.index_ = 0

    def __setitem__(self, key, item):
        utils.add_no_duplicate(self.elements, key, item)

    def __getitem__(self, key):
        return self.elements[key]

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements.items())

    def subset(self, element_type: Element):
        element_subset = ElementSet()
        for element in self.elements.values():
            if isinstance(element, element_type):
                element_subset[element.name] = element
        return element_subset

    def get_element_type(self, element_type):
        element_list = []
        for element in self.elements.values():
            if type(element).__name__ == element_type:
                element_list.append(element)
        return element_list

    @property
    def element_types(self) -> list:
        types = []
        for element in self.elements.values():
            types.append(type(element).__name__)
        return list(set(types))

    @property
    def element_counts(self) -> dict:
        types = []
        for element in self.elements.values():
            types.append(type(element).__name__)
        return dict(Counter(types))

    @property
    def gages(self):
        gages = {}
        for name, element in self.elements.items():
            if "Observed Hydrograph Gage" in element.attrs.keys():
                gages[name] = element.attrs["Observed Hydrograph Gage"]
        return gages
