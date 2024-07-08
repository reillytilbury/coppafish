import napari
import itertools
from typing import Optional, List

from .. import log
from ..utils import tiles_io
from ..setup import Notebook


def view_extracted_images(
    nb: Notebook,
    tiles: Optional[List[int]] = None,
    rounds: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
) -> None:
    """
    View the extracted images located at `nb.file_names.tile_unfiltered_dir`.

    Args:
        nb (Notebook): notebook.
        tiles (Optional[List[int]], optional): tiles to view. Default: all tiles.
        rounds (Optional[List[int]], optional): rounds to view. Default: all rounds.
        channels (Optional[List[int]], optional): channels to view. Default: all channels.
    """
    assert nb.has_page("extract"), "Extract must be run first"

    if tiles is None:
        tiles = nb.basic_info.use_tiles.copy()
    if rounds is None:
        rounds = nb.basic_info.use_rounds.copy()
    if channels is None:
        channels = nb.basic_info.use_channels.copy()

    viewer = napari.Viewer(title="Coppafish extracted images")

    for t, r, c in itertools.product(tiles, rounds, channels):
        file_path = nb.file_names.tile_unfiltered[t][r][c]
        if not tiles_io.image_exists(file_path):
            log.warn(f"Image at {file_path} not found, skipping")
            continue
        image_trc = tiles_io._load_image(file_path)
        viewer.add_image(image_trc, name=f"{t=}, {r=}, {c=}")

    napari.run()
