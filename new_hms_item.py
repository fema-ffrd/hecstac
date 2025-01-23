from hecstac.hms.item import HMSItem

if __name__ == "__main__":

    hms_project_file = "hms_model.hms"  # the HMS project file (.hms)
    href = "hms_model.json"  # The STAC item json to be created

    hms_item = HMSItem(hms_project_file, href)

    hms_item.save_object()
