from hecstac.common.logger import initialize_logger
from hecstac.hms.item import HMSModelItem

if __name__ == "__main__":
    initialize_logger()
    hms_project_file = "/Users/slawler/Downloads/duwamish/Duwamish_SST.hms"
    href = hms_project_file.replace(".hms", ".json")

    hms_item = HMSModelItem(hms_project_file, href)
    hms_item.save_object()
