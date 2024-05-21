import os
import time
import json
import shutil
from typing import Any, Dict, List, Tuple

import zarr
import numpy as np

from .. import utils


class NotebookPage:
    _variables: Dict[str, Any] = {}

    _page_name: str
    _page_name_key: str = "page_name"
    _time_created: float
    _time_created_key: str = "time_created"
    _version: str
    _version_key: str = "version"

    # Each page variable is given a list. The list contains a datatype(s) in the first index followed by a description.
    # A variable can be allowed to take multiple datatypes by separating them with an ' or '. Check the supported
    # types by looking at the function _is_types at the end of this file. The 'tuple' is a special datatype that can be
    # nested. For example, tuple[tuple[int]] is a valid datatype.
    _datatype_separator: str = " or "
    _datatype_nest_start: str = "["
    _datatype_nest_end: str = "]"
    _options: Dict[str, Dict[str, list]] = {
        "basic_info": {
            "anchor_channel": ["int or none", "Channel in anchor used. None if anchor not used."],
            "anchor_round": [
                "int or none",
                "Index of anchor round (typically the first round after imaging rounds so `anchor_round = n_rounds`)."
                + "`None` if anchor not used.",
            ],
            "dapi_channel": [
                "int or none",
                "Channel in anchor round that contains *DAPI* images. `None` if no *DAPI*.",
            ],
            "use_channels": ["tuple[int]", "n_use_channels. Channels in imaging rounds to use throughout pipeline."],
            "use_rounds": ["tuple[int]", "n_use_rounds. Imaging rounds to use throughout pipeline."],
            "use_z": ["tuple[int]", "z planes used to make tile *npy* files"],
            "use_tiles": [
                "tuple[int]",
                "n_use_tiles tiles to use throughout pipeline."
                + "For an experiment where the tiles are arranged in a $4 \\times 3$ ($n_y \\times n_x$) grid, "
                + "tile indices are indicated as below:"
                + "\n"
                + "| 2  | 1  | 0  |"
                + "\n"
                + "| 5  | 4  | 3  |"
                + "\n"
                + "| 8  | 7  | 6  |"
                + "\n"
                + "| 11 | 10 | 9  |",
            ],
            "use_dyes": ["tuple[int]", "n_use_dyes dyes to use when assigning spots to genes."],
            "dye_names": [
                "tuple[str] or none",
                "Names of all dyes so for gene with code $360...$,"
                + "gene appears with `dye_names[3]` in round $0$, `dye_names[6]` in round $1$, `dye_names[0]` in round $2$ etc."
                + "`none` if each channel corresponds to a different dye.",
            ],
            "is_3d": [
                "bool",
                "`True` if *3D* pipeline used, `False` if *2D*",
            ],
            "channel_camera": [
                "tuple[int] or none",
                "`channel_camera[i]` is the wavelength in *nm* of the camera on channel $i$."
                + "`none` if `dye_names = none`.",
            ],
            "channel_laser": [
                "tuple[int] or none",
                "`channel_laser[i]` is the wavelength in *nm* of the laser on channel $i$."
                + "`none` if `dye_names = none`.",
            ],
            "tile_pixel_value_shift": [
                "int",
                "This is added onto every tile (except *DAPI*) when it is saved and removed from every tile when loaded."
                + "Required so we can have negative pixel values when save to *npy* as *uint16*."
                + "*Typical=15000*",
            ],
            "n_extra_rounds": [
                "int",
                "Number of non-imaging rounds, typically 1 if using anchor and 0 if not.",
            ],
            "n_rounds": [
                "int",
                "Number of imaging rounds in the raw data",
            ],
            "tile_sz": [
                "int",
                "$yx$ dimension of tiles in pixels",
            ],
            "n_tiles": [
                "int",
                "Number of tiles in the raw data",
            ],
            "n_channels": [
                "int",
                "Number of channels in the raw data",
            ],
            "nz": [
                "int",
                "Number of z-planes used to make the *npy* tile images (can be different from number in raw data).",
            ],
            "n_dyes": [
                "int",
                "Number of dyes used",
            ],
            "tile_centre": [
                "ndarray[float]",
                "`[y, x, z]` location of tile centre in units of `[yx_pixels, yx_pixels, z_pixels]`."
                + "For *2D* pipeline, `tile_centre[2] = 0`",
            ],
            "tilepos_yx_nd2": [
                "ndarray[int]",
                "[n_tiles x 2] `tilepos_yx_nd2[i, :]` is the $yx$ position of tile with *fov* index $i$ in the *nd2* file."
                + "Index 0 refers to `YX = [0, 0]`"
                + "Index 1 refers to `YX = [0, 1]` if `MaxX > 0`",
            ],
            "tilepos_yx": [
                "ndarray[int]",
                "[n_tiles x 2] `tilepos_yx[i, :]` is the $yx$ position of tile with tile directory (*npy* files) index $i$."
                + "Equally, `tilepos_yx[use_tiles[i], :]` is $yx$ position of tile `use_tiles[i]`."
                + "Index 0 refers to `YX = [MaxY, MaxX]`"
                + "Index 1 refers to `YX = [MaxY, MaxX - 1]` if `MaxX > 0`",
            ],
            "pixel_size_xy": [
                "float",
                "$yx$ pixel size in microns",
            ],
            "pixel_size_z": [
                "float",
                "$z$ pixel size in microns",
            ],
            "use_anchor": [
                "bool",
                "whether or not to use anchor",
            ],
            "use_preseq": [
                "bool",
                "whether or not to use pre-seq round",
            ],
            "pre_seq_round": [
                "int or none",
                "round number of pre-seq round",
            ],
            "bad_trc": [
                "tuple[tuple[int]]",
                "Tuple of bad tile, round, channel combinations. If a tile, round, channel combination is in this,"
                + "it will not be used in the pipeline.",
            ],
            "software_version": [
                "str",
                "Coppafish version, as given by *coppafish/_version.py*",
            ],
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
    _type_prefixes: Dict[str, str] = {
        "int": "json",
        "float": "json",
        "str": "json",
        "bool": "json",
        "tuple": "json",
        "none": "json",
        "ndarray": "npz",
        "zarr": "zarr",
    }

    def __init__(self, page_name: str) -> None:
        self._sanity_check_variable_options()
        if page_name not in self._options.keys():
            raise ValueError(f"Could not find _variable_options for page called {page_name}")
        self._page_name = page_name
        self._time_created = time.time()
        self._version = utils.system.get_software_version()

    def save(self, page_directory: str, /) -> None:
        if os.path.isdir(page_directory):
            raise SystemError(f"Found existing page directory at {page_directory}")
        if len(self._get_unset_variables()) > 0:
            raise ValueError(
                f"Cannot save unfinished page. Variable(s) {self._get_unset_variables()} not assigned yet."
            )

        os.mkdir(page_directory)
        metadata_path = self._get_metadata_path(page_directory)
        self._save_metadata(metadata_path)
        for name, value in self._variables.items():
            type_as_str: str = self._options[self._page_name][name][0]
            self._save_variable(name, value, type_as_str, page_directory)

    def load(self, page_directory: str, /) -> None:
        """
        Load all variables from inside the given directory. All variables already set inside of the page are
        overwritten.
        """
        if not os.path.isdir(page_directory):
            raise FileNotFoundError(f"Could not find page directory at {page_directory} to load from")
        for name in self._options[self._page_name].keys():
            self._variables[name] = self._load_variable(name, page_directory)

        metadata_path = self._get_metadata_path(page_directory)
        self._load_metadata(metadata_path)
        for name in self._options[self._page_name].keys():
            self._variables[name] = self._load_variable(name, page_directory)

    def __gt__(self, other: str) -> None:
        """
        Handles getting a variable's description by doing `notebook_page > "variable_name"`.
        """
        assert type(other) is str
        if other not in self._options[self._page_name].keys():
            print(f"No variable named {other}")
            return
        print(f"{self._options[self._page_name][other][1]}")

    def _get_unset_variables(self) -> Tuple[str]:
        """
        Returns a tuple of all variable names that have not been set to a valid value in the notebook page.
        """
        unset_variables = []
        for variable_name in self._options[self._page_name].keys():
            if variable_name not in self._variables.keys():
                unset_variables.append(variable_name)
        return tuple(unset_variables)

    def __setattr__(self, name: str, value: Any, /) -> None:
        """
        Deals with syntax `notebook_page.name = value`.
        """
        if name not in self._options[self._page_name].keys():
            raise NameError(f"Cannot set variable {name} in {self._page_name} page. It is not inside _variable_options")
        expected_types = self._get_expected_types(name)
        if not self._is_types(value, expected_types):
            raise TypeError(f"Failed to set variable {name} to type {type(value)}. Expected type(s) {expected_types}")
        self._variables[name] = value

    def __getattribute__(self, name: str, /) -> Any:
        """
        Deals with syntax `value = notebook_page.name`.
        """
        if name not in self._options[self._page_name].keys():
            raise NameError(f"Variable {name} in {self._page_name} page is not inside _variable_options")
        if name not in self._variables.keys():
            raise ValueError(f"Variable {name} in {self._page_name} page has not been set")
        return self._variables[name]

    def _save_metadata(self, file_path: str) -> None:
        assert not os.path.isfile(file_path), f"Metadata file at {file_path} should not exist"

        metadata = {
            self._page_name_key: self._page_name,
            self._time_created_key: self._time_created,
            self._version_key: self._version,
        }
        with open(file_path, "x") as file:
            file.write(json.dumps(metadata, indent=4))

    def _load_metadata(self, file_path: str) -> None:
        assert os.path.isfile(file_path), f"Metadata file at {file_path} not found"

        metadata: dict = None
        with open(file_path, "r") as file:
            metadata = json.loads(file.read)
            assert type(metadata) is dict
        self._page_name = metadata[self._page_name_key]
        self._time_created = metadata[self._time_created_key]
        self._version = metadata[self._version_key]

    def _get_metadata_path(self, page_directory: str) -> str:
        return os.path.join(page_directory, "metadata.json")

    def _get_page_directory(self, in_directory: str) -> str:
        return os.path.join(in_directory, self._page_name)

    def _get_expected_types(self, name: str) -> str:
        return self._options[self._page_name][name][0]

    def _save_variable(self, name: str, value: Any, type_as_str: str, page_directory: str) -> None:
        file_prefix = self._type_str_to_prefix(type_as_str)
        file_path = os.path.join(page_directory, f"{name}.{file_prefix}")

        if file_prefix == "json":
            with open(file_path, "x") as file:
                file.write(json.dumps({"value": value}, indent=4))
        elif file_prefix == "npz":
            value.setflags(write=False)
            np.savez_compressed(file_path, value)
        elif file_prefix == "zarr":
            if not os.path.isdir(value):
                raise FileNotFoundError(f"Failed to find zarr at {value}")
            value_zarr: zarr.Array = zarr.open(value)
            if type(value_zarr) is not zarr.Array:
                raise TypeError(f"File at {file_path} was of type {type(value_zarr)}, expected zarr array")
            saved_value = zarr.create(
                path=file_path,
                mode="w-",
                shape=value.shape,
                dtype=value.dtype,
                order=value.order,
                compressor=value.compressor,
                chunks=value.chunks,
                zarr_version=2,
            )
            saved_value[:] = value_zarr[:]
            saved_value.read_only = True
            del value_zarr
            if os.path.normpath(value) != os.path.normpath(file_path):
                # Delete the old location of the zarr array.
                shutil.rmtree(value)
        else:
            raise NotImplementedError(f"File prefix {file_prefix} is not supported")

    def _load_variable(self, name: str, page_directory: str) -> Any:
        types_as_str = self._options[self._page_name][name].split(self._datatype_separator)
        file_prefix = self._type_str_to_prefix(types_as_str[0])
        file_path = os.path.join(page_directory, f"{name}.{file_prefix}")

        if file_prefix == "json":
            with open(file_path, "r") as file:
                value = json.loads(file.read())["value"]
            return value
        elif file_prefix == "npz":
            return np.load(file_path)["arr_0"]
        elif file_prefix == "zarr":
            return file_path
        else:
            raise NotImplementedError(f"File prefix {file_prefix} is not supported")

    def _sanity_check_variable_options(self) -> None:
        # Only multiple datatypes can be options for the same variable if they save to the same save file type. So, a
        # variable's type cannot be "ndarray[int] or zarr" because they save into different file types.
        for name, types_as_str in self._options.items():
            unique_prefixes = set()
            for type_as_str in types_as_str.split(self._datatype_separator):
                unique_prefixes.add(self._type_str_to_prefix(type_as_str))
            if len(unique_prefixes) > 1:
                raise TypeError(
                    f"Variable {name} has incompatible types: {' and '.join(unique_prefixes)} in _variable_options"
                )

    def _type_str_to_prefix(self, type_as_str: str) -> str:
        return self._type_prefixes[type_as_str.split(self._datatype_nest_start)[0]]

    def _is_types(self, value: Any, types_as_str: str) -> bool:
        valid_types: List[str] = types_as_str.split(self._datatype_separator)
        for type_str in valid_types:
            if self._is_types(value, type_str):
                return True
        return False

    def _is_type(self, value: Any, type_as_str: str) -> bool:
        assert (
            self._datatype_separator not in type_as_str
        ), f"Type {type_as_str} cannot contain the phrase {self._datatype_separator}"

        if type_as_str == "none":
            return value is None
        elif type_as_str == "int":
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
                    if not self._is_type(subvalue, type_as_str[len("tuple") : -len(self._datatype_nest_end)]):
                        return False
                return True
        elif type_as_str == "ndarray[float]":
            return type(value) is np.ndarray and value.dtype.type is np.float_
        elif type_as_str == "ndarray[int]":
            return type(value) is np.ndarray and value.dtype.type is np.int_
        elif type_as_str == "ndarray[bool]":
            return type(value) is np.ndarray and value.dtype.type is np.bool_
        elif type_as_str == "zarr":
            # A zarr is specified by pointing the notebook page to the location of the zarr array. This way the array
            # does not need to be in memory all the time and the array can be deleted once it is saved within the
            # notebook page.
            return type(value) is str and os.path.isdir(value) and value.endswith(".zarr")
        else:
            raise TypeError(f"Unexpected type '{type_as_str}' found in _variable_options in NotebookPage")
