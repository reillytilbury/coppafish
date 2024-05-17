import json
import os
import time

import numpy as np

from .. import log, utils


# The notebook page will have file structure:
# notebook_page/
# ├─ metadata.json
# ├─ other_data.any
#
# where all custom variable data can be saved in any way.
class NotebookPage:
    _created_time: float
    _name: str
    _software_version: str

    # Unlikely to change variables.
    _metadata_name: str = "metadata.json"
    _json_indent: int = 4
    _created_time_key: str = "created_time"
    _name_key: str = "name"
    _software_version_key: str = "version"

    def __init__(self) -> None:
        self._created_time = time.time()
        self._software_version = utils.system.get_software_version()

    def save(self, directory: str, overwrite: bool = False) -> None:
        assert type(directory) is str
        path = str(os.path.join(directory, self._name))
        if not overwrite:
            assert not (os.path.isdir(path) or os.path.isfile(path)), f"Notebook page {path} already exists."
        elif os.path.isdir(path):
            log.warn(f"Found existing notebook page at {path}. Any existing data files may be overwritten.")

        self._save_metadata(os.path.join(directory, self._metadata_name))

    def load(self, directory: str) -> None:
        """
        Load all variables saved to disk at directory into self.
        """
        assert type(directory) is str
        assert os.path.isdir(directory), f"Notebook page path {directory} not found"
        metadata_path = os.path.join(directory, self._metadata_name)
        assert os.path.isfile(metadata_path), f"metadata file at {metadata_path} not found"

        self._load_metadata(metadata_path)

    def _verify_types(self) -> None:
        """
        Check the types of each notebook page variable.
        """
        assert type(self._created_time) is float
        assert type(self._name) is str
        assert type(self._software_version) is str

    def _save_metadata(self, file_path: str) -> None:
        assert type(file_path) is str

        metadata = dict()
        metadata[self._name_key] = self._name
        metadata[self._created_time_key] = self._created_time
        metadata[self._software_version_key] = self._software_version
        with open(file_path, "w") as file:
            file.write(json.dumps(metadata, indent=self._json_indent))

    def _load_metadata(self, file_path: str) -> None:
        assert type(file_path) is str

        with open(file_path, "r") as file:
            metadata: dict = json.loads(file.read())
        assert type(metadata) is dict, f"File {file_path} must be a dictionary json"

        self._created_time = metadata[self._created_time_key]
        self._name = metadata[self._name_key]
        self._software_version = metadata[self._software_version_key]

    def _assert_int(self, variable) -> None:
        assert type(variable) is int

    def _assert_optional_int(self, variable) -> None:
        assert variable is None or type(variable) is int

    def _assert_float(self, variable) -> None:
        assert type(variable) is float

    def _assert_bool(self, variable) -> None:
        assert type(variable) is bool

    def _assert_str(self, variable) -> None:
        assert type(variable) is str

    def _assert_optional_str(self, variable) -> None:
        assert variable is None or type(variable) is str

    def _assert_list_of_type(self, variable, of_type: type) -> None:
        assert type(variable) is list
        for value in variable:
            assert type(value) is of_type

    def _assert_ndarray_of_type(self, variable, of_type: type) -> None:
        assert type(variable) is np.ndarray
        assert isinstance(variable.dtpe.type(), of_type)
