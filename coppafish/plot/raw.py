import napari
import numbers
import numpy as np
from tqdm import tqdm
from typing import Union, Optional, List, Tuple
from ..utils import raw, nd2
from ..setup import Notebook
from ..pipeline.basic_info import set_basic_info


def add_basic_info_no_save(nb: Notebook):
    """
    This adds the `basic_info` page to the notebook without saving the notebook.

    Args:
        nb: Notebook with no `basic_info` page.

    """
    if not nb.has_page("basic_info"):
        nb._no_save_pages["basic_info"] = {}  # don't save if add basic_info page
        config = nb.get_config()
        nbp_basic = set_basic_info(config["file_names"], config["basic_info"])
        nb += nbp_basic


def get_raw_images(
    nb: Notebook, tiles: List[int], rounds: List[int], channels: List[int], use_z: List[int]
) -> np.ndarray:
    """
    This loads in raw images for the experiment corresponding to the *Notebook*.

    Args:
        nb: Notebook for experiment
        tiles: npy (as opposed to nd2 fov) tile indices to view.
            For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as
            below:

            | 2  | 1  | 0  |

            | 5  | 4  | 3  |

            | 8  | 7  | 6  |

            | 11 | 10 | 9  |
        rounds: Rounds to view.
        channels: Channels to view.
        use_z: Which z-planes to load in from raw data.

    Returns:
        `raw_images` - `[len(tiles) x len(rounds) x len(channels) x n_y x n_x x len(use_z)]` uint16 array.
        `raw_images[t, r, c]` is the `[n_y x n_x x len(use_z)]` image for tile `tiles[t]`, round `rounds[r]` and channel
        `channels[c]`.
    """
    n_tiles = len(tiles)
    n_rounds = len(rounds)
    n_channels = len(channels)
    n_images = n_rounds * n_tiles * n_channels
    ny = nb.basic_info.tile_sz
    nx = ny
    nz = len(use_z)

    raw_images = np.zeros((n_tiles, n_rounds, n_channels, ny, nx, nz), dtype=np.uint16)
    with tqdm(total=n_images) as pbar:
        pbar.set_description(f"Loading in raw data")
        for r in range(n_rounds):
            round_dask_array, _ = raw.load_dask(nb.file_names, nb.basic_info, r=rounds[r])
            # TODO: Can get rid of these two for loops, when round_dask_array is always a dask array.
            #  At the moment though, is not dask array when using nd2_reader (On Mac M1).
            for t in range(n_tiles):
                for c in range(n_channels):
                    pbar.set_postfix({"round": rounds[r], "tile": tiles[t], "channel": channels[c]})
                    raw_images[t, r, c] = raw.load_image(
                        nb.file_names, nb.basic_info, tiles[t], channels[c], round_dask_array, rounds[r], use_z
                    )
                    pbar.update(1)
    return raw_images


def number_to_list(var_list: List) -> Tuple:
    # Converts every value in variables to a list if it is a single number
    # Args:
    #     var_list: List of variables which need converting to list
    #
    # Returns:
    #     var_list with variables converted into list.

    for i in range(len(var_list)):
        if isinstance(var_list[i], numbers.Number):
            var_list[i] = [var_list[i]]
    return tuple(var_list)


def view_raw(
    nb: Optional[Notebook] = None,
    tiles: Union[int, List[int]] = 0,
    rounds: Union[int, List[int]] = 0,
    channels: Optional[Union[int, List[int]]] = None,
    use_z: Optional[Union[int, List[int]]] = None,
    config_file: Optional[str] = None,
):
    """
    Function to view raw data in napari.
    There will upto 4 scrollbars for each image to change tile, round, channel and z-plane.

    !!! warning "Requires access to `nb.file_names.input_dir`"

    Args:
        nb: *Notebook* for experiment. If no *Notebook* exists, pass `config_file` instead.
        tiles: npy (as opposed to nd2 fov) tile indices to view.
            For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as
            below:

            | 2  | 1  | 0  |

            | 5  | 4  | 3  |

            | 8  | 7  | 6  |

            | 11 | 10 | 9  |
        rounds: rounds to view (`anchor` will be `nb.basic_info.n_rounds` i.e. the last round.)
        channels: Channels to view. If `None`, will load all channels.
        use_z: Which z-planes to load in from raw data. If `None`, will use load all z-planes (except from first
            one if `config['basic_info']['ignore_first_z_plane'] == True`).
        config_file: path to config file for experiment.
    """
    if nb is None:
        nb = Notebook(config_file=config_file)
    add_basic_info_no_save(nb)  # deal with case where there is no notebook yet
    if channels is None:
        channels = np.arange(nb.basic_info.n_channels)
    if use_z is None:
        use_z = nb.basic_info.use_z
    tiles, rounds, channels, use_z = number_to_list([tiles, rounds, channels, use_z])

    raw_images = get_raw_images(nb, tiles, rounds, channels, use_z)
    viewer = napari.Viewer()
    viewer.add_image(np.moveaxis(raw_images, -1, 3), name="Raw Images")

    @viewer.dims.events.current_step.connect
    def update_slider(event):
        viewer.status = (
            f"Tile: {tiles[event.value[0]]}, Round: {rounds[event.value[1]]}, "
            f"Channel: {channels[event.value[2]]}, Z: {use_z[event.value[3]]}"
        )

    viewer.dims.axis_labels = ["Tile", "Round", "Channel", "z", "y", "x"]
    viewer.dims.set_point([0, 1, 2], [0, 0, 0])  # set to first tile, round and channel initially
    napari.run()


def view_tile_layout(
    nb: Notebook,
    num_rotations: int = 0,
    tiles: Optional[Union[int, List[int]]] = None,
    channel: int = 0
):
    """
    Function to view the tile layout in napari. Images will be middle z plane from nd2 files in the anchor round.
    Channel can be specified.
    Args:
        nb: Notebook containing at least basic info and file names.
        num_rotations: Number of 90 degree rotations to apply to each individual tile. These rotations always in the
        direction taking the y axis to the x axis.
        tiles: Which tiles to view. If `None`, will view use_tiles.
        channel: Which channel to view.
    """
    assert channel in nb.basic_info.use_channels or channel == nb.basic_info.dapi_channel, f"Invalid channel: {channel}"

    if tiles is None:
        tiles = nb.basic_info.use_tiles

    raw_images = get_raw_images(
        nb,
        tiles=tiles,
        rounds=[nb.basic_info.anchor_round],
        channels=[channel],
        use_z=[nb.basic_info.nz // 2],
    )[:, 0, 0, :, :, 0]

    # First rotate the images. This makes num_rotations rotations in the direction taking the y axis to the x axis
    raw_images = np.rot90(raw_images, k=num_rotations, axes=(1, 2))
    tiles_nd2 = nd2.get_nd2_tile_ind(tiles, nb.basic_info.tilepos_yx_nd2, nb.basic_info.tilepos_yx)
    tilepos_yx = nb.basic_info.tilepos_yx.copy()
    tilepos_yx_nd2 = nb.basic_info.tilepos_yx_nd2.copy()

    # Now plot
    expected_overlap = nb.get_config()["stitch"]["expected_overlap"]
    tile_sz = nb.basic_info.tile_sz
    yx_step = tile_sz * (1 - expected_overlap)
    viewer = napari.Viewer()
    text_npy_ind = {
        "string": [f"{t}" for t in tiles],
        "color": "red",
        "size": 36,
    }
    text_nd2_ind = {
        "string": [f"{t}" for t in tiles_nd2],
        "color": "cyan",
        "size": 36,
    }
    text_npy_pos = {
        "string": [f"{tilepos_yx[t]}" for t in tiles],
        "color": "red",
        "size": 36,
    }
    text_nd2_pos = {
        "string": [f"{tilepos_yx_nd2[t]}" for t in tiles_nd2],
        "color": "cyan",
        "size": 36,
    }
    point_locs = [tilepos_yx[t] * yx_step + np.array([tile_sz // 2, tile_sz // 2]) for t in tiles]
    for t in range(len(tiles)):
        viewer.add_image(raw_images[t], name=f"Tile {tiles[t]}", translate=tilepos_yx[tiles[t]] * yx_step)
    viewer.add_points(point_locs, text=text_npy_ind, size=0, name="NPY Tile Indices")
    viewer.add_points(point_locs, text=text_nd2_ind, size=0, name="ND2 Tile Indices", visible=False)
    viewer.add_points(point_locs, text=text_npy_pos, size=0, name="NPY YX Positions", visible=False)
    viewer.add_points(point_locs, text=text_nd2_pos, size=0, name="ND2 YX Positions", visible=False)
    viewer.add_vectors(
        np.array([[0, 0], [0, 1]]) * tile_sz, edge_width=50, edge_color="green", name="Y Axis")
    viewer.add_vectors(
        np.array([[0, 0], [1, 0]]) * tile_sz, edge_width=50, edge_color="green",  name="X Axis"
    )
    axis_text = {
        "string": ["Y", "X"],
        "color": "green",
        "size": 36,
    }
    axis_label_locs = np.array([[tile_sz, -200], [-200, tile_sz]])
    viewer.add_points(axis_label_locs, text=axis_text, size=0, name="Axis Labels")
    viewer.dims.axis_labels = ["y", "x"]

    napari.run()
