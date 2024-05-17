import json
import os
from typing import List, Optional

from .base import NotebookPage


class FileNames(NotebookPage):
    _name: str = "file_names"

    _data_name: str = "data.json"

    input_dir: str
    output_dir: str
    tile_dir: str
    tile_unfiltered_dir: str
    round: List[str]
    anchor: Optional[str]
    raw_extension: str
    raw_metadata: Optional[str]
    dye_camera_laser: str
    code_book: str
    scale: str
    spot_details_info: str
    psf: str
    big_dapi_image: Optional[str]
    big_anchor_image: str
    pciseq: List[str]
    tile: List[List[List[str]]]
    tile_unfiltered: List[List[List[str]]]
    fluorescent_bead_path: Optional[str]
    preseq: Optional[str]
    initial_bleed_matrix: Optional[str]

    def save(self, directory: str) -> None:
        self._verify_types()
        NotebookPage.save(self, directory)
        basic_info_data = {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "tile_dir": self.tile_dir,
            "tile_unfiltered_dir": self.tile_unfiltered_dir,
            "round": self.round,
            "anchor": self.anchor,
            "raw_extension": self.raw_extension,
            "raw_metadata": self.raw_metadata,
            "dye_camera_laser": self.dye_camera_laser,
            "code_book": self.code_book,
            "scale": self.scale,
            "spot_details_info": self.spot_details_info,
            "psf": self.psf,
            "big_dapi_image": self.big_dapi_image,
            "big_anchor_image": self.big_anchor_image,
            "pciseq": self.pciseq,
            "tile": self.tile,
            "tile_unfiltered": self.tile_unfiltered,
            "fluorescent_bead_path": self.fluorescent_bead_path,
            "preseq": self.preseq,
            "initial_bleed_matrix": self.initial_bleed_matrix,
        }
        with open(os.path.join(directory, self._data_name), "w") as file:
            file.write(json.dumps(basic_info_data, indent=NotebookPage._json_indent))

    def load(self, directory: str) -> None:
        NotebookPage.load(self, directory)
        basic_info_data = dict()
        with open(os.path.join(directory, self._data_name), "r") as file:
            basic_info_data: dict = json.loads(file.read())
        assert type(basic_info_data) is dict

        self.input_dir = basic_info_data["input_dir"]
        self.output_dir = basic_info_data["output_dir"]
        self.tile_dir = basic_info_data["tile_dir"]
        self.tile_unfiltered_dir = basic_info_data["tile_unfiltered_dir"]
        self.round = basic_info_data["round"]
        self.anchor = basic_info_data["anchor"]
        self.raw_extension = basic_info_data["raw_extension"]
        self.raw_metadata = basic_info_data["raw_metadata"]
        self.dye_camera_laser = basic_info_data["dye_camera_laser"]
        self.code_book = basic_info_data["code_book"]
        self.scale = basic_info_data["scale"]
        self.spot_details_info = basic_info_data["spot_details_info"]
        self.psf = basic_info_data["psf"]
        self.big_dapi_image = basic_info_data["big_dapi_image"]
        self.big_anchor_image = basic_info_data["big_anchor_image"]
        self.pciseq = basic_info_data["pciseq"]
        self.tile = basic_info_data["tile"]
        self.tile_unfiltered = basic_info_data["tile_unfiltered"]
        self.fluorescent_bead_path = basic_info_data["fluorescent_bead_path"]
        self.preseq = basic_info_data["preseq"]
        self.initial_bleed_matrix = basic_info_data["initial_bleed_matrix"]

        self._verify_types()

    def _verify_types(self) -> None:
        NotebookPage._verify_types(self)

        NotebookPage._assert_str(self, self.input_dir)
        NotebookPage._assert_str(self, self.output_dir)
        NotebookPage._assert_str(self, self.tile_dir)
        NotebookPage._assert_str(self, self.tile_unfiltered_dir)
        NotebookPage._assert_list_of_type(self, self.round, str)
        NotebookPage._assert_optional_str(self, self.anchor)
        NotebookPage._assert_str(self, self.raw_extension)
        NotebookPage._assert_optional_str(self, self.raw_metadata)
        NotebookPage._assert_str(self, self.dye_camera_laser)
        NotebookPage._assert_str(self, self.code_book)
        NotebookPage._assert_str(self, self.scale)
        NotebookPage._assert_str(self, self.spot_details_info)
        NotebookPage._assert_str(self, self.psf)
        NotebookPage._assert_optional_str(self, self.big_dapi_image)
        NotebookPage._assert_str(self, self.big_anchor_image)
        NotebookPage._assert_list_of_type(self, self.pciseq, str)
        NotebookPage._assert_list_of_type(self, self.tile, list)
        NotebookPage._assert_list_of_type(self, self.tile_unfiltered, list)
        NotebookPage._assert_optional_str(self, self.fluorescent_bead_path)
        NotebookPage._assert_optional_str(self, self.preseq)
        NotebookPage._assert_optional_str(self, self.initial_bleed_matrix)
