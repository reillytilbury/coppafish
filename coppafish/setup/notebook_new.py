import os
import time
import json
from typing import List

from .. import log, utils
from .notebook_page_new import NotebookPage


class Notebook:
    _directory: str

    _time_created: float
    _time_created_key: str = "time_created"
    _version: str
    _version_key: str = "version"

    _pages: List[NotebookPage]
    _options = {
        "basic_info": [
            "*basic_info* page contains information that is used at all stages of the pipeline.",
        ],
        "file_names": [
            "*file_names* page contains all files that are used throughout the pipeline.",
        ],
        "extract": [
            "*extract* page contains information related to extraction of raw input files for use in coppafish."
        ],
        "filter": [
            "*filter* page contains information on image filtering applied to extracted images.",
        ],
        "filter_debug": [
            "*filter_debug* page contains additional information on filtering that is not used later in the pipeline."
        ],
        "find_spots": [
            "*find_spots* page contains information about spots found on every tiles, rounds and channels.",
        ],
        "stitch": [
            "*stitch* page contains information about how tiles were stitched together to produce global coords."
        ],
        "register": [
            "*register* page contains best found solutions to allign images.",
        ],
        "register_debug": [
            "*register_debug* page contains information on how the image allignments in *register* were calculated."
        ],
        "ref_spots": [
            "*ref_spots* page contains gene assignments and info for spots found on reference round.",
        ],
        "call_spots": [
            "*call_spots* page contains `bleed_matrix` and expected code for each gene.",
        ],
        "omp": [
            "*omp* page contains gene assigments and information for spots found through Orthogonal Matching Pursuit."
        ],
        "thresholds": [
            "*thresholds* page contains quality thresholds which affect which spots plotted and exported to pciSeq."
        ],
    }

    def __init__(self, notebook_dir: str) -> None:
        """
        Load the notebook found at the given directory. Or, if the directory does not exist, create the directory.
        """
        self._directory = notebook_dir
        self._time_created = time.time()
        self._version = utils.system.get_software_version()
        if not os.path.isdir(self._directory):
            log.info(f"Creating new notebook at {self._directory}")
            os.mkdir(self._directory)
            self._save_metadata(self._directory)
        self._load_metadata(self._directory)
        self._pages = []
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
            self._pages.append(loaded_page)

    def __gt__(self, page_name: str) -> None:
        """
        Handles printing a page's description by doing `notebook > "page_name"`.
        """
        assert type(page_name) is str
        if page_name not in self._options.keys():
            print(f"No page named {page_name}")
            return
        print(f"{self._options[page_name][0]}")

    def _save_metadata(self) -> None:
        assert os.path.isdir(self._directory)
        file_path = self._get_metadata_path()
        assert not os.path.isfile(file_path), f"Notebook metadata at {file_path} already exists"

        metadata = {
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
        self._version = metadata[self._version_key]
        self._time_created = metadata[self._time_created_key]

    def _get_metadata_path(self) -> str:
        return os.path.join(self._directory, "metadata.json")
