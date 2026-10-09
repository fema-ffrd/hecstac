# hecstac

[![CI](https://github.com/fema-ffrd/hecstac/actions/workflows/main-push.yaml/badge.svg?branch=main)](https://github.com/fema-ffrd/hecstac/actions/workflows/main-push.yaml)
[![Dev CI](https://github.com/fema-ffrd/hecstac/actions/workflows/dev-push.yaml/badge.svg?branch=dev)](https://github.com/fema-ffrd/hecstac/actions/workflows/dev-push.yaml)
[![Documentation Status](https://readthedocs.org/projects/hecstac/badge/?version=latest)](https://hecstac.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/hecstac.svg)](https://badge.fury.io/py/hecstac)
[![Docker Scout](https://github.com/fema-ffrd/hecstac/actions/workflows/docker-scout.yaml/badge.svg)](https://github.com/fema-ffrd/hecstac/actions/workflows/docker-scout.yaml)

Utilities for creating STAC items from HEC models

**hecstac** is an open-source Python library designed to mine metadata from HEC model simulations for use in the development of catalogs documenting probabilistic flood studies. This project automates the generation of STAC Items and Assets from HEC-HMS and HEC-RAS model files, enabling improved data and metadata management.

## Installation

This package may be installed using pip with the following command

```
$ pip install hecstac
```

> [!NOTE]
> Fiona, used by the HEC-HMS readers, does not currently publish Python 3.14 wheels. Installing hecstac on Python 3.14 therefore requires GDAL development files, including `gdal-config` (`libgdal-dev` on Ubuntu).

FFRD calibration checks are optional:

```
$ pip install "hecstac[ffrd-calibration-check]"
```

The `ffrd-calibration-check` extra is available on Python 3.10 through 3.13. The current `rasqc` release pins `obstore==0.6.0`, which has no CPython 3.14 build; the extra will support Python 3.14 once rasqc publishes a compatible dependency set.

## FFRD

While `hecstac` was created principally in support of FFRD pilot projects, the ability to create STAC based metadata items for HEC models (RAS and HMS in particular) has guided some design and implementation decisions that make it flexible enough to support more generalized use cases.

There will be modules and workflows that are very specifically designed for FFRD, and those will be generally distinguishable via _ffrd_ in the name of the file / class function / etc. The Dockerfiles that are included in the repo are designed specifically in support of these pilots, and as such are not meant for general “out-of-the-box” use cases. For specifics on FFRD use cases please see the documentation.

## Examples

For some example workflows, please [read the docs](https://hecstac.readthedocs.io/en/latest/user_guide.html#workflows).
