from hecstac.hms.item import HMSItem

if __name__ == "__main__":

    hms_project_file = r"C:\Users\mdeshotel\OneDrive - Dewberry\Projects\FFRD_STAC\duwamish\Duwamish_SST.hms"  # the HMS project file (.hms)
    href = "Duwamish_SST.json"  # The STAC item json to be created

    hms_item = HMSItem(hms_project_file, href)

    hms_item.save_object()
