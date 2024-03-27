import napari
import itertools
from typing import Optional, List

from .. import logging
from ..utils import tiles_io
from ..setup.notebook import Notebook


def view_filtered_images(
    nb: Notebook,
    tiles: Optional[List[int]] = None,
    rounds: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
) -> None:
    """
    View the filtered images located at `nb.file_names.tile_dir`.

    Args:
        nb (Notebook): notebook.
        tiles (Optional[List[int]], optional): tiles to view. Default: all tiles.
        rounds (Optional[List[int]], optional): rounds to view. Default: all rounds.
        channels (Optional[List[int]], optional): channels to view. Default: all channels.
    """
    assert nb.has_page("filter"), "Filter must be run first"

    if tiles is None:
        tiles = nb.basic_info.use_tiles.copy()
    if rounds is None:
        rounds = nb.basic_info.use_rounds.copy()
    if channels is None:
        channels = nb.basic_info.use_channels.copy()

    viewer = napari.Viewer(title="Coppafish filtered images")

    for t, r, c in itertools.product(tiles, rounds, channels):
        file_path = nb.file_names.tile_unfiltered[t][r][c]
        if not tiles_io.image_exists(file_path, nb.extract.file_type):
            logging.warn(f"Image at {file_path} not found, skipping")
            continue
        image_trc = tiles_io.load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t, r, c)
        viewer.add_image(image_trc, name=f"{t=}, {r=}, {c=}")

    napari.run()
