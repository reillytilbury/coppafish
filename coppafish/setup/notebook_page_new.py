import os
from typing import Any, Dict

import numpy as np


class NotebookPage:
    # Each page variable is given a list. The list contains a datatype(s) in the first index followed by a description.
    # A variable can be allowed to take multiple datatypes by separating them with an ' or '.
    _page_name: str
    _datatype_separator: str = " or "
    _to_save: Dict[str, Any] = {}
    _variable_options: dict = {
        "basic_info": {
            "anchor_channel": ["int or none", "Channel in anchor used. None if anchor not used."],
        },
        "file_names": {},
        "extract": {},
        "filter": {},
        "filter_debug": {},
        "find_spots": {},
        "stitch": {},
        "register": {},
        "register_debug": {},
        "ref_spots": {},
        "call_spots": {},
        "omp": {},
        "thresholds": {},
    }

    def __init__(self, page_name: str) -> None:
        self._page_name = page_name

    def save(self, directory: str) -> None:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Could not find directory at {directory} to load from")
        page_directory = self._page_directory(directory)
        if os.path.isdir(page_directory):
            raise SystemError(f"Found existing page directory at {page_directory}")

    def load(self, directory: str) -> None:
        """
        Load all variables from inside the given directory. All variables already added to the page instance are
        overwritten.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Could not find directory at {directory} to load from")
        page_directory = self._page_directory(directory)
        if not os.path.isdir(page_directory):
            raise FileNotFoundError(f"Could not find page directory at {page_directory} to load from")

    def __setattr__(self, name: str, value: Any, /) -> None:
        """
        Deals with syntax `notebook_page.name = value`
        """
        self._to_save[name] = value

    def _page_directory(self, in_directory: str) -> str:
        return os.path.join(in_directory, self._page_name)

    def _is_type(self, value: Any, type_as_str: str) -> bool:
        if type_as_str == "int":
            return type(value) is int
        elif type_as_str == "str":
            return type(value) is str
        elif type_as_str == "bool":
            return type(value) is bool
        elif type_as_str == "tuple":
            return type(value) is tuple
        elif type_as_str.startswith("tuple"):
            if len(value) == 0:
                return True
            else:
                for subvalue in value:
                    if not self._is_type(subvalue, type_as_str[5:-1]):
                        return False
                return True
        elif type_as_str == "ndarray[float]":
            return type(value) is np.ndarray and value.dtype.type is np.float_
        elif type_as_str == "ndarray[int]":
            return type(value) is np.ndarray and value.dtype.type is np.int_
        else:
            raise TypeError(f"Unexpected type '{type_as_str}' found in _variable_options in NotebookPage")
