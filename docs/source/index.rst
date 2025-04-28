hecstac
=======
.. toctree::
   :maxdepth: 2
   :hidden:

   User Guide <user_guide.rst>
   FFRD Containers <ffrd_containers.rst>
   API Reference <modules.rst>


.. note::
   This version is tagged as **testing**. Features and APIs are unstable and subject to change.

:code:`hecstac` is an open-source Python library designed to access and process `HEC <https://www.hec.usace.army.mil/software/>`_ model data to create catalogs, metadata,
and data products for use in managing simulations performed probabilistic modeling campaigns in the cloud. This project automates the generation of STAC items from `HEC-HMS <https://www.hec.usace.army.mil/software/hec-hms/>`_  and
`HEC-RAS <https://www.hec.usace.army.mil/software/hec-ras/>`_  models providing a consistent and interoperable data format for sharing and publishing results.


**Key Features:**

   - Parses HEC-HMS and HEC-RAS model files.
   - Generates STAC Items and Assets for model data.
   - Creates Thumbnails from model geometry data.
   - Compiles multiple model data outputs from event based modeling.


Data Sharing and Publishing
--------------------------------

hecstac facilitates FAIR data principles by enabling:

   - Exporting items and assets with published metadata terms.
   - Publishing STAC items directly to a **STAC API**.
