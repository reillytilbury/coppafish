import json
import os
import time
from typing import Any, Tuple

from .. import log, utils
from .notebook_page import NotebookPage


class Notebook:
    _directory: str

    # Attribute names allowed to be set inside the notebook page that are not in _options.
    _VALID_ATTRIBUTE_NAMES = ("config_path", "_config_path", "_directory", "_time_created", "_version")

    _metadata_name: str = "_metadata.json"

    _time_created: float
    _time_created_key: str = "time_created"
    _version: str
    _version_key: str = "version"

    _config_path: str
    _config_path_key: str = "config_path"

    def get_config_path(self) -> str:
        return self._config_path

    def set_config_path(self, path: str) -> str:
        assert type(path) is str
        if self._config_path is not None:
            raise ValueError(f"Cannot set config_path, already set to {self._config_path}")

        self._config_path = path

    config_path = property(get_config_path, set_config_path)

    _options = {
        "basic_info": [
            "*basic_info* page contains information that is used at all stages of the pipeline.",
        ],
        "file_names": [
            "*file_names* page contains all files that are used throughout the pipeline.",
        ],
        "extract": [
            "*extract* page contains information related to extraction of raw input files for use in coppafish.",
        ],
        "filter": [
            "*filter* page contains information on image filtering applied to extracted images.",
        ],
        "filter_debug": [
            "*filter_debug* page contains additional information on filtering that is not used later in the pipeline.",
        ],
        "find_spots": [
            "*find_spots* page contains information about spots found on every tiles, rounds and channels.",
        ],
        "stitch": [
            "*stitch* page contains information about how tiles were stitched together to produce global coords.",
        ],
        "register": [
            "*register* page contains best found solutions to allign images.",
        ],
        "register_debug": [
            "*register_debug* page contains information on how the image allignments in *register* were calculated.",
        ],
        "ref_spots": [
            "*ref_spots* page contains gene assignments and info for spots found on reference round.",
        ],
        "call_spots": [
            "*call_spots* page contains `bleed_matrix` and expected code for each gene.",
        ],
        "omp": [
            "*omp* page contains gene assigments and information for spots found through Orthogonal Matching Pursuit.",
        ],
        "thresholds": [
            "*thresholds* page contains quality thresholds which affect which spots plotted and exported to pciSeq.",
        ],
        "debug": [
            "*debug* page for unit testing.",
        ],
    }

    def __init__(self, notebook_dir: str, /) -> None:
        """
        Load the notebook found at the given directory. Or, if the directory does not exist, create the directory.
        """
        assert type(notebook_dir) is str

        self._config_path = None
        self._directory = notebook_dir
        self._time_created = time.time()
        self._version = utils.system.get_software_version()
        if not os.path.isdir(self._directory):
            log.info(f"Creating notebook at {self._directory}")
            os.mkdir(self._directory)
            self._save()
        self._load()
        # Check directory for existing notebook pages and load them in.
        for page_name in os.listdir(self._directory):
            if page_name == self._metadata_name:
                continue
            page_path = os.path.join(self._directory, page_name)
            if os.path.isfile(page_path):
                raise FileExistsError(f"Unexpected file {page_path} inside the notebook")
            if page_name not in self._options.keys():
                raise IsADirectoryError(f"Unexpected directory at {page_path} inside the notebook")
            loaded_page = NotebookPage(page_name)
            loaded_page.load(page_path)
            self.__setattr__(page_name, loaded_page)

    def __iadd__(self, page: NotebookPage):
        """
        Add and save a new page to the notebook using syntax notebook += notebook_page.
        """
        if type(page) is not NotebookPage:
            raise TypeError(f"Cannot add type {type(page)} to the notebook")
        unset_variables = page.get_unset_variables()
        if len(unset_variables) > 0:
            raise ValueError(
                f"Page {page.name} must have every variable set before adding to the notebook. "
                + f"Variable(s) unset: {', '.join(unset_variables)}"
            )

        self.__setattr__(page.name, page)
        self._save()
        return self

    def has_page(self, page_name: str) -> bool:
        assert type(page_name) is str
        if page_name not in self._options.keys():
            raise ValueError(f"Not a real page name: {page_name}. Expected one of {', '.join(self._options.keys())}")

        try:
            self.__getattribute__(page_name)
            return True
        except AttributeError:
            return False

    def resave(self) -> None:
        """
        Delete the notebook on disk and re-save every page using the instance in memory.
        """
        # NOTE: This function should not be used by the coppafish pipeline. This is purely a function for developers
        # to manually change variables that are already saved to disk. Even then, this function should be used as
        # little as possible as it will inevitably cause bugs.
        start_time = time.time()
        for page in self._get_existing_pages():
            page.resave(self._get_page_directory(page.name))
        os.remove(self._get_metadata_path())
        self._save_metadata()
        end_time = time.time()
        print(f"Notebook re-saved in {end_time - start_time:.2f}s")

    def get_unqiue_versions(self) -> Tuple[str]:
        """
        Get the unique software versions found inside of the notebook and pages.
        """
        unique_versions = set()
        unique_versions.add(self._version)
        for page in self._get_existing_pages():
            unique_versions.add(page.version)
        return tuple(unique_versions)

    def __setattr__(self, name: str, value: Any, /) -> None:
        """
        Deals with syntax `notebook.name = value`.
        """
        if name in self._VALID_ATTRIBUTE_NAMES:
            object.__setattr__(self, name, value)
            return

        if type(value) is not NotebookPage:
            raise TypeError(f"Can only add NotebookPage classes to the Notebook, got {type(value)}")
        if self.has_page(value.name):
            raise ValueError(f"Notebook already contains page named {value.name}")

        object.__setattr__(self, name, value)

    def __gt__(self, page_name: str) -> None:
        """
        Print a page's description by doing `notebook > "page_name"`.
        """
        assert type(page_name) is str

        if page_name not in self._options.keys():
            print(f"No page named {page_name}")
            return
        print(f"{self._options[page_name][0]}")

    def _save(self) -> None:
        """
        Save the notebook to the directory specified when the notebook was instantiated.
        """
        start_time = time.time()
        self._save_metadata()
        for page in self._get_existing_pages():
            page_dir = self._get_page_directory(page.name)
            page.save(page_dir)
        end_time = time.time()
        log.info(f"Notebook saved in {end_time - start_time:.2f}s")

    def _load(self) -> None:
        self._load_metadata()

    def _get_existing_pages(self) -> Tuple[NotebookPage]:
        pages = []
        for page_name in self._options.keys():
            if self.has_page(page_name):
                pages.append(self.__getattribute__(page_name))
        return tuple(pages)

    def _get_page_directory(self, page_name: str) -> str:
        assert type(page_name) is str

        return str(os.path.join(self._directory, page_name))

    def _save_metadata(self) -> None:
        assert os.path.isdir(self._directory)

        file_path = self._get_metadata_path()
        if os.path.isfile(file_path):
            return
        metadata = {
            self._config_path_key: self._config_path,
            self._time_created_key: self._time_created,
            self._version_key: self._version,
        }
        with open(file_path, "x") as file:
            file.write(json.dumps(metadata, indent=4))

    def _load_metadata(self) -> None:
        assert os.path.isdir(self._directory)
        file_path = self._get_metadata_path()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Could not find notebook metadata at {file_path}")

        metadata = dict()
        with open(file_path, "r") as file:
            metadata = json.loads(file.read())
        self._config_path = metadata[self._config_path_key]
        self._version = metadata[self._version_key]
        self._time_created = metadata[self._time_created_key]

    def _get_metadata_path(self) -> str:
        return os.path.join(self._directory, self._metadata_name)
