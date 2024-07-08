import itertools
from typing import List, Optional

import napari

from ..setup import Notebook


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
        image_trc = nb.filter.images[t, r, c]
        viewer.add_image(image_trc, name=f"{t=}, {r=}, {c=}")

    napari.run()
