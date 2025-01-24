from hecstac.ras.item import RasModelItem

prj_file = "ElkMiddle/ElkMiddle.prj"
item_href = "folder/item.json"
item_id = "ElkMiddle"
crs = "EPSG:4326"

model_item = RasModelItem(prj_file=prj_file, crs=crs, item_id=item_id, href=item_href)
model_item.populate()
# model_item.get_usgs_data(True)

model_item.add_model_thumbnail(True, True, ["mesh_areas", "breaklines", "bc_lines"])
model_item.save_object()
