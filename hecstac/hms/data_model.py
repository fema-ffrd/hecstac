"""HEC-HMS STAC Item data classes."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

import hecstac.hms.utils as utils


@dataclass
class Element:
    """Parent class of basin elements (Subbasins, Reaches, etc)."""

    name: str
    attrs: OrderedDict


@dataclass
class BasinHeader:
    """Header of .basin."""

    attrs: dict


@dataclass
class BasinLayerProperties:
    """Part of footer of .basin, find via 'Basin Layer Properties:'. Data is stored as a series of layers rather than a set of attributes, so just storing the raw content for now."""

    content: str


@dataclass
class Control(Element):
    """Represents a control element."""

    pass


@dataclass
class Grid(Element):
    """Represents a grid element."""

    pass


@dataclass
class Precipitation(Element):
    """Represents a precipitation element."""

    pass


@dataclass
class Temperature(Element):
    """Represents a temperature element."""

    pass


@dataclass
class ET(Element):
    """Represents a ET element."""

    pass


@dataclass
class Subbasin_ET(Element):
    """Represents a Subbasin_ET element."""

    pass


@dataclass
class Gage(Element):
    """Represents a gage element."""

    pass


@dataclass
class ComputationPoints:
    """Part of footer of .basin, find via 'Computation Points:'. Data has some complex attributes with nested end-flags, so just storing raw content for now."""

    content: str


@dataclass
class BasinSpatialProperties:
    """Part of footer of .basin, find via 'Basin Spatial Properties:'. Data has some complex attributes with nested end-flags, so just storing raw content for now."""

    content: str


@dataclass
class BasinSchematicProperties:
    """Part of footer of .basin, find via 'Basin Schematic Properties:'."""

    attrs: dict


@dataclass
class Run:
    """Runs contained in the .run file."""

    name: str
    attrs: dict


@dataclass
class Subbasin(Element):
    """Represents a Subbasin element."""

    geom: Polygon = None


@dataclass
class Table(Element):
    """Represents a Table element."""

    pass


@dataclass
class Pattern(Element):
    """Represents a Pattern element."""

    pass


@dataclass
class Reach(Element):
    """Represents a Reach element."""

    geom: LineString = None
    slope: float = None  # assumed units of the coordinate system is the same as what is used for the project.. need to confirm this assumption


@dataclass
class Junction(Element):
    """Represents a Junction element."""

    geom: Point = None


@dataclass
class Sink(Element):
    """Represents a Sink element."""

    geom: Point = None


@dataclass
class Reservoir(Element):
    """Represents a Reservoir element."""

    geom: Point = None


@dataclass
class Source(Element):
    """Represents a Source element."""

    geom: Point = None


@dataclass
class Diversion(Element):
    """Represents a Diversion element."""

    geom: Point = None


class ElementSet:
    """Behaves like a dictionary of Basin elements (Subbasins, Reaches, etc) with key conflict checking."""

    def __init__(self):
        self.elements: dict[str, Element] = {}
        self.index_ = 0

    def __setitem__(self, key, item):
        """Add an element to the set."""
        utils.add_no_duplicate(self.elements, key, item)

    def __getitem__(self, key):
        """Retrieve an element by name."""
        return self.elements[key]

    def __len__(self):
        """Return the number of elements."""
        return len(self.elements)

    def __iter__(self):
        """Iterate over elements."""
        return iter(self.elements.items())

    def subset(self, element_type: Element):
        """Retrieve a subset of elements of a given type."""
        element_subset = ElementSet()
        for element in self.elements.values():
            if isinstance(element, element_type):
                element_subset[element.name] = element
        return element_subset

    def get_element_type(self, element_type):
        """Retrieve elements of a specific type by name."""
        element_list = []
        for element in self.elements.values():
            if type(element).__name__ == element_type:
                element_list.append(element)
        return element_list

    @property
    def element_types(self) -> list:
        """Get a list of unique element types."""
        types = []
        for element in self.elements.values():
            types.append(type(element).__name__)
        return list(set(types))

    @property
    def element_counts(self) -> dict:
        """Get a count of each element type."""
        types = []
        for element in self.elements.values():
            types.append(type(element).__name__)
        return dict(Counter(types))

    @property
    def gages(self):
        """Retrieve gage elements with their observed hydrograph gage names."""
        gages = {}
        for name, element in self.elements.items():
            if "Observed Hydrograph Gage" in element.attrs.keys():
                gages[name] = element.attrs["Observed Hydrograph Gage"]
        return gages
