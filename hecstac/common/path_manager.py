from pathlib import Path


class LocalPathManager:
    """
    Builds consistent paths for STAC items and collections assuming a top level local catalog
    """

    def __init__(self, model_root_dir: str):
        self._model_root_dir = model_root_dir

    @property
    def model_root_dir(self) -> str:
        return str(self._model_root_dir)

    @property
    def model_parent_dir(self) -> str:
        return str(Path(self._model_root_dir).parent)

    def item_path(self, item_id: str) -> str:
        return f"{self._model_root_dir}/{item_id}.json"

    def derived_item_asset(self, filename: str) -> str:
        return f"{self._model_root_dir}/{filename}"
