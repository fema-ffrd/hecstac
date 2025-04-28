===============================
Ras Calibration Check Container
===============================

.. code-block:: bash
    # Pull Container from GH Repository
    docker pull ghcr.io/fema-ffrd/hecstac:0.1.0rc4-dev

Overview
========

The current container hosts a single python script that scans s3, creates a STAC metadata Item (with geometry thumbnail) and performs
a calibration check. The current calibration check supports only a review of file and element names based on the SOP naming conventions.
It leverages the `rasqc`, `rashdf`, and `hecstac` Python packages.

**Note**: To support concurrent development, some dependencies may be built in the container as the PyPi versions required may not be available.

Features
========

- **GDAL Base**: Built on top of a minimal Ubuntu image with GDAL support for geospatial data processing.
- **Python Environment**: Utilizes Python virtual environments for package isolation.
- **Custom Script**: Includes a custom Python script for the calibration check process, tailored for S3-based models.
- **Configurable Input**: Accepts configuration via JSON strings or files to define S3 buckets and prefixes.

Installation
============

This container is based on the `ghcr.io/osgeo/gdal:ubuntu-small-latest` image and utilizes Python 3. The necessary Python packages and dependencies are installed during the build process.

To use this container, ensure you have Docker installed and run the following command:

.. code-block:: bash

    docker build -t hecstac .

Usage
=====

The main script requires environment variables for AWS credentials and a configuration file or string. It can be run with Docker using:

.. code-block:: bash

    docker run --rm -e AWS_ACCESS_KEY_ID=<your-access-key> -e AWS_SECRET_ACCESS_KEY=<your-secret-key> \
        hecstac --config '<your-config-json>'

Configuration
=============

The script requires a configuration in JSON format, which specifies the S3 bucket and prefix for the HEC-RAS models. The configuration can be provided as a JSON string or as a path to a JSON file.

Example JSON Configurations
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

Environment Variables
=====================

- **AWS_ACCESS_KEY_ID**: Your AWS access key ID.
- **AWS_SECRET_ACCESS_KEY**: Your AWS secret access key.

These environment variables are required for accessing the S3 resources.

Development
===========

The container is built in two stages:
1. **Build Stage**: Compiles and installs the `rasqc` and `hecstac` packages.
2. **Production Stage**: Sets up a clean environment with only the necessary runtime dependencies.
