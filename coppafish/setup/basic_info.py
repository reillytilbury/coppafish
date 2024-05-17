import json
import os
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .base import NotebookPage


class BasicInfo(NotebookPage):
    _name: str = "basic_info"

    _data_name: str = "data.json"

    anchor_channel: Optional[int]
    anchor_round: Optional[int]
    dapi_channel: Optional[int]
    use_channels: List[int]
    use_rounds: List[int]
    use_z: List[int]
    use_tiles: List[int]
    use_dyes: List[int]
    dye_names: List[str]
    is_3d: bool
    channel_camera: List[int]
    channel_laser: List[int]
    tile_pixel_value_shift: int
    n_extra_rounds: int
    n_rounds: int
    tiles_sz: int
    n_tiles: int
    n_channels: int
    n_dyes: int
    tile_centre: npt.NDArray[np.float_]
    tilepos_yx_nd2: npt.NDArray[np.int_]
    tilepos_yx: npt.NDArray[np.int_]
    pixel_size_xy: float
    pixel_size_z: float
    use_anchor: bool
    use_preseq: bool
    preseq_round: Optional[int]
    bad_trc: List[Tuple[int, int, int]]

    def save(self, directory: str) -> None:
        self._verify_types()
        NotebookPage.save(self, directory)
        basic_info_data = {
            "anchor_channel": self.anchor_channel,
            "anchor_round": self.anchor_round,
            "dapi_channel": self.dapi_channel,
            "use_channels": self.use_channels,
            "use_rounds": self.use_rounds,
            "use_z": self.use_z,
            "use_tiles": self.use_tiles,
            "use_dyes": self.use_dyes,
            "dye_names": self.dye_names,
            "is_3d": self.is_3d,
            "channel_camera": self.channel_camera,
            "channel_laser": self.channel_laser,
            "tile_pixel_value_shift": self.tile_pixel_value_shift,
            "n_extra_rounds": self.n_extra_rounds,
            "n_rounds": self.n_rounds,
            "tiles_sz": self.tiles_sz,
            "n_tiles": self.n_tiles,
            "n_channels": self.n_channels,
            "n_dyes": self.n_dyes,
            "tile_centre": self.tile_centre,
            "tilepos_yx_nd2": self.tilepos_yx_nd2,
            "tilepos_yx": self.tilepos_yx,
            "pixel_size_xy": self.pixel_size_xy,
            "pixel_size_z": self.pixel_size_z,
            "use_anchor": self.use_anchor,
            "use_preseq": self.use_preseq,
            "preseq_round": self.preseq_round,
            "bad_trc": self.bad_trc,
        }
        with open(os.path.join(directory, self._data_name), "w") as file:
            file.write(json.dumps(basic_info_data, indent=NotebookPage._json_indent))

    def load(self, directory: str) -> None:
        NotebookPage.load(self, directory)
        basic_info_data = dict()
        with open(os.path.join(directory, self._data_name), "r") as file:
            basic_info_data: dict = json.loads(file.read())
        assert type(basic_info_data) is dict

        self.anchor_channel = basic_info_data["anchor_channel"]
        self.anchor_round = basic_info_data["anchor_round"]
        self.dapi_channel = basic_info_data["dapi_channel"]
        self.use_channels = basic_info_data["use_channels"]
        self.use_rounds = basic_info_data["use_rounds"]
        self.use_z = basic_info_data["use_z"]
        self.use_tiles = basic_info_data["use_tiles"]
        self.use_dyes = basic_info_data["use_dyes"]
        self.dye_names = basic_info_data["dye_names"]
        self.is_3d = basic_info_data["is_3d"]
        self.channel_camera = basic_info_data["channel_camera"]
        self.channel_laser = basic_info_data["channel_laser"]
        self.tile_pixel_value_shift = basic_info_data["tile_pixel_value_shift"]
        self.n_extra_rounds = basic_info_data["n_extra_rounds"]
        self.n_rounds = basic_info_data["n_rounds"]
        self.tiles_sz = basic_info_data["tiles_sz"]
        self.n_tiles = basic_info_data["n_tiles"]
        self.n_channels = basic_info_data["n_channels"]
        self.n_dyes = basic_info_data["n_dyes"]
        self.tile_centre = basic_info_data["tile_centre"]
        self.tilepos_yx_nd2 = basic_info_data["tilepos_yx_nd2"]
        self.tilepos_yx = basic_info_data["tilepos_yx"]
        self.pixel_size_xy = basic_info_data["pixel_size_xy"]
        self.pixel_size_z = basic_info_data["pixel_size_z"]
        self.use_anchor = basic_info_data["use_anchor"]
        self.use_preseq = basic_info_data["use_preseq"]
        self.preseq_round = basic_info_data["preseq_round"]
        self.bad_trc = basic_info_data["bad_trc"]

        self._verify_types()

    def _verify_types(self) -> None:
        NotebookPage._verify_types(self)

        NotebookPage._assert_optional_int(self, self.anchor_channel)
        NotebookPage._assert_optional_int(self, self.anchor_round)
        NotebookPage._assert_optional_int(self, self.dapi_channel)
        NotebookPage._assert_list_of_type(self, self.use_channels, int)
        NotebookPage._assert_list_of_type(self, self.use_rounds, int)
        NotebookPage._assert_list_of_type(self, self.use_z, int)
        NotebookPage._assert_list_of_type(self, self.use_tiles, int)
        NotebookPage._assert_list_of_type(self, self.use_dyes, str)
        NotebookPage._assert_list_of_type(self, self.dye_names, str)
        NotebookPage._assert_bool(self, self.is_3d)
        NotebookPage._assert_list_of_type(self, self.channel_camera, int)
        NotebookPage._assert_list_of_type(self, self.channel_laser, int)
        NotebookPage._assert_int(self, self.tile_pixel_value_shift)
        NotebookPage._assert_int(self, self.n_extra_rounds)
        NotebookPage._assert_int(self, self.n_rounds)
        NotebookPage._assert_int(self, self.tiles_sz)
        NotebookPage._assert_int(self, self.n_tiles)
        NotebookPage._assert_int(self, self.n_channels)
        NotebookPage._assert_int(self, self.n_dyes)
        NotebookPage._assert_ndarray_of_type(self, self.tile_centre, np.float_)
        NotebookPage._assert_ndarray_of_type(self, self.tilepos_yx_nd2, np.int_)
        NotebookPage._assert_ndarray_of_type(self, self.tilepos_yx, np.int_)
        NotebookPage._assert_float(self, self.pixel_size_xy)
        NotebookPage._assert_float(self, self.pixel_size_z)
        NotebookPage._assert_bool(self, self.use_anchor)
        NotebookPage._assert_bool(self, self.use_preseq)
        NotebookPage._assert_optional_int(self, self.preseq_round)
        NotebookPage._assert_list_of_type(self, self.bad_trc, tuple)
