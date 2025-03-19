# hecstac
[![CI](https://github.com/fema-ffrd/hecstac/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/fema-ffrd/hecstac/actions/workflows/ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/hecstac/badge/?version=latest)](https://hecstac.readthedocs.io/en/latest/?badge=latest)
[![Release](https://github.com/fema-ffrd/hecstac/actions/workflows/release.yaml/badge.svg)](https://github.com/fema-ffrd/hecstac/actions/workflows/release.yaml)
[![PyPI version](https://badge.fury.io/py/hecstac.svg)](https://badge.fury.io/py/hecstac)

Utilities for creating STAC items from HEC models

**hecstac** is an open-source Python library designed to mine metadata from HEC model simulations for use in the development of catalogs documenting probabilistic flood studies. This project automates the generation of STAC Items and Assets from HEC-HMS and HEC-RAS model files, enabling improved data and metadata management.

## Developer Setup
Create a virtual environment in the project directory:
```
$ python -m venv venv
```

Activate the virtual environment:
```
# For macOS/Linux
$ source ./venv/bin/activate
(venv) $

# For Windows
> ./venv/Scripts/activate
```

Install dev dependencies:
```
(venv) $ pip install ".[dev]"
```

***Testing HEC-RAS model item creation***

- Download the HEC-RAS example project data from USACE. The data can be downloaded [here](https://github.com/HydrologicEngineeringCenter/hec-downloads/releases/download/1.0.33/Example_Projects_6_6.zip).
- In 'new_ras_item.py', set the ras_project_file to the path of the 2D Muncie project file (ex. ras_project_file = "Example_Projects_6_6/2D Unsteady Flow Hydraulics/Muncie/Muncie.prj").
- For projects that have projection information within the geometry .hdf files, the CRS info can automatically be detected. The Muncie data lacks that projection info so it must be set by extracting the WKT projection string and setting the CRS in 'new_ras_item.py' to the projection string. The projection can be found in the Muncie/GIS_Data folder in Muncie_IA_Clip.prj.
- Once the CRS and project file location have been set, a new item can be created with 'python -m new_ras_item' in the command line. The new item will be added inside the model directory at the same level as the project file.
