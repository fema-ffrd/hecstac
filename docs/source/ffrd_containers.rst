===============================
Ras Calibration Check Container
===============================

Overview
========

The current container hosts a variety of python scripts to support the creation of STAC Items for HEC-RAS models stored in S3 buckets. It leverages the `rasqc`, `rashdf`, and `hecstac` Python packages.

Current workflow scripts include:

- **ffrd_ras_conformance_item.py**: Creates STAC Items for HEC-RAS conformance event models and associated parquet time series assets containing flow at boundary and reference lines and stage at reference points, reference lines, and boundary condition lines.
- **ffrd_ras_calibration_item.py**: Creates STAC Items for HEC-RAS calibration models which include thumbnails showing model geometry features and parquets of geospatial data.
- **ffrd_ras_event_item.py**: Creates STAC Items for HEC-RAS calibration event models. Also creates parquets with various time series data for boundary conditions, reference points, and reference lines.
- **ras_calibration_check.py**: Performs naming convention QC checks on HEC-RAS models utilizing the `rasqc` library.

Features
========

- **GDAL Base**: Built on top of a minimal Ubuntu image with GDAL support for geospatial data processing.
- **Python Environment**: Utilizes Python virtual environments for package isolation.
- **Custom Script**: Includes various custom Python scripts, tailored for S3-based models.
- **Configurable Input**: Accepts configuration via JSON strings or files to define S3 buckets and prefixes. Example configuration files are provided for each script.

Installation
============

This container is based on the `ghcr.io/osgeo/gdal:ubuntu-small-latest` image and utilizes Python 3. The necessary Python packages and dependencies are installed during the build process.

To use this container, ensure you have Docker installed and run the following commands:

.. code-block:: bash

    docker pull ghcr.io/fema-ffrd/hecstac:latest

    docker run -it --rm ghcr.io/fema-ffrd/hecstac:latest

Usage
=====

All scripts require environment variables for AWS credentials which can be set in a .env file using the following command:

.. code-block:: bash

    printf "AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY>\nAWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY>\nAWS_REGION=us-east-1\n" > .env

The processes can then be run with the desired script and configuration:

.. code-block:: bash

    python ffrd_ras_conformance_item.py --config example-configs/ras_conformance_item.json

Configuration
=============

The scripts requires a configuration in JSON format, which specifies the inputs and resulting outputs. The configuration can be provided as a JSON string or as a path to a JSON file.

Example RAS Calibration Check JSON Configuration
---------------------------

Single model example:

.. code-block:: json

    {
        "bucket": "trinity-pilot",
        "prefix": "calibration/hydraulics/bridgeport",
        "skip_hdf_files": false
    }

Multiple model example:

.. code-block:: json

    [
        {
            "bucket": "trinity-pilot",
            "prefix": "calibration/hydraulics/bridgeport",
            "skip_hdf_files": true
        },
        {
            "bucket": "trinity-pilot",
            "prefix": "calibration/hydraulics/kickapoo",
            "ras_model_name": "trinity_1203_kickapoo",
            "skip_hdf_files": true
        }
    ]

Example RAS Conformance Item JSON Configuration
---------------------------

Single model example:

.. code-block:: json

    {
        "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork",
        "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=blw-clear-fork/flow_timeseries.pq",
        "stage_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=blw-clear-fork/stage_timeseries.pq"
    }

Multiple model example:

.. code-block:: json

    [
        {
            "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/blw-clear-fork",
            "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=blw-clear-fork/flow_timeseries.pq",
            "stage_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=blw-clear-fork/stage_timeseries.pq"
        },
        {
            "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/bardwell-creek",
            "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=bardwell-creek/flow_timeseries.pq",
            "stage_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=bardwell-creek/stage_timeseries.pq"
        },
        {
            "model_prefix": "s3://trinity-pilot/conformance/simulations/event-data/1/hydraulics/bedias-creek",
            "flow_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=bedias-creek/flow_timeseries.pq",
            "stage_output_path": "s3://trinity-pilot/stac/prod-support/conformance/event_id=1/ras_model=bedias-creek/stage_timeseries.pq"
        }
    ]

Development
===========

The container is built in two stages:

1. **Build Stage**: Compiles and installs the `hecstac` package.
2. **Production Stage**: Sets up a clean environment with only the necessary runtime dependencies.
