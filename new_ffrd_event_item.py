from pystac import Item

from hecstac.common.logger import initialize_logger
from hecstac.events.ffrd import FFRDEventItem

if __name__ == "__main__":
    initialize_logger()
    hms_source_model_item_path = "/Users/slawler/Downloads/duwamish/Duwamish_SST.json"
    hms_source_model_item = Item.from_file(hms_source_model_item_path)

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

    realization = "R01"
    block_group = "BG01"
    event_id = "E01"

    ffrd_event_item_id = f"{realization}-{block_group}-{event_id}"
    dest_href = f"/Users/slawler/Downloads/duwamish/{ffrd_event_item_id}.json"

    ffrd_event_item = FFRDEventItem(
        realization=realization,
        block_group=block_group,
        event_id=event_id,
        source_model_items=[hms_source_model_item],
        hms_simulation_files=hms_simulation_files,
    )

    ffrd_event_item.save_object(dest_href=dest_href)
    # TODO: check hrefs for links, may need to sanitize
