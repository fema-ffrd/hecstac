"""Creates a STAC Item from an event."""

from pystac import Item

from hecstac.events.logger import initialize_logger
from hecstac.events.ffrd import EventItem

if __name__ == "__main__":
    initialize_logger()

    # HMS Info
    hms_source_model_item_path = "/Users/slawler/Downloads/duwamish/Duwamish_SST.json"
    hms_source_model_item = Item.from_file(hms_source_model_item_path)
    # TODO: create a function to read these in
    hms_simulation_files = [
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/Duwamish_SST.basin",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/Duwamish_SST.grid",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/SST.control",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/SST_normalized.dss",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/SST_normalized.met",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/SST_normalized_ExportedPrecip.dss",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/SST_normalized_",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/simulations/validation/100/Hydrology/Duwamish_SST/data/Storm.dss",
    ]

    # RAS Info
    ras_source_model_item_path = "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/checkpoint-validation/hydraulics/duwamish-20250106/Duwamish_17110013.json"
    ras_source_model_item = Item.from_file(ras_source_model_item_path)
    ras_simulation_files = [
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/scenario-simple-levees/simulations/1/hydraulics/duwamish-v6.6/rasoutput.log",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/scenario-simple-levees/simulations/1/hydraulics/duwamish-v6.6/Duwamish_17110013.p01.hdf",
        "/Users/slawler/Downloads/model-library-2/ffrd-duwamish/scenario-simple-levees/simulations/1/hydraulics/duwamish-v6.6/Duwamish_17110013.b01",
    ]

    # Event Info
    realization = "R01"
    block_group = "BG01"
    event_id = "E01"

    ffrd_event_item_id = f"{realization}-{block_group}-{event_id}"
    dest_href = f"/Users/slawler/Downloads/duwamish/{ffrd_event_item_id}.json"

    ffrd_event_item = EventItem(
        realization=realization,
        block_group=block_group,
        event_id=event_id,
        source_model_items=[
            hms_source_model_item
        ],  # TODO: when ras geom is fixed, add link to ras_source_model_item_path
        hms_simulation_files=hms_simulation_files,
        ras_simulation_files=ras_simulation_files,
    )

    ffrd_event_item.save_object(dest_href=dest_href)
    # TODO: check hrefs for links, may need to sanitize
