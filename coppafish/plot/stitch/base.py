import matplotlib.pyplot as plt
import napari
import numpy as np

from ...setup import Notebook
from tqdm import tqdm


def view_stitch_checkerboard(nb: Notebook, downsample_factor_yx: int = 4):
    """
    Load in tiles and view them in global coords with a green and red checkerboard pattern.
    Args:
        nb: Notebook (must have page stitch)
        downsample_factor_yx: amount to downsample the tiles by in y and x
    """
    # load in frequently used variables
    use_tiles = list(nb.basic_info.use_tiles)
    tilepos_yx = nb.basic_info.tilepos_yx[use_tiles]
    tile_origin = nb.stitch.tile_origin[use_tiles] - np.min(nb.stitch.tile_origin[use_tiles], axis=0)
    downsample_vector = np.array([downsample_factor_yx, downsample_factor_yx, 1])
    tile_origin = tile_origin // downsample_vector
    # convert tile_origin from yxz to zyx
    tile_origin = tile_origin[:, [2, 0, 1]]
    mid_z, tile_size = nb.basic_info.nz // 2, nb.basic_info.tile_sz
    yxz_ind = np.ix_(
        np.arange(0, tile_size, downsample_factor_yx),
        np.arange(0, tile_size, downsample_factor_yx),
        np.arange(mid_z - 10, mid_z + 10, 1),
    )
    tiles = []
    viewer = napari.Viewer()

    # Load in the tiles
    for t in tqdm(use_tiles, total=len(use_tiles), desc="Loading tiles"):
        tile = nb.filter.images[t, nb.basic_info.anchor_round, nb.basic_info.dapi_channel]
        tile = tile[yxz_ind]
        # convert tile from yxz to zyx
        tile = np.moveaxis(tile, -1, 0)
        tile = tile[:, ::downsample_factor_yx, ::downsample_factor_yx]
        tiles.append(tile)

    # Create the checkerboard pattern
    for i, t in enumerate(use_tiles):
        y, x = tilepos_yx[i]
        colour = 'red' if (y + x) % 2 == 0 else 'green'
        viewer.add_image(tiles[i], name=f'tile_{t}', translate=tile_origin[i], blending='additive', colormap=colour)

    napari.run()


def view_shifts(nb: Notebook):
    """
    View the shifts of each tile as a heatmap.
    Args:
        nb: Notebook (must have page stitch)
    """
    tilepos_yx = nb.basic_info.tilepos_yx
    use_tiles = nb.basic_info.use_tiles
    n_rows, n_cols = np.max(tilepos_yx, axis=0) + 1

    # Create the heatmap
    shift_heatmap = np.zeros((n_rows, n_cols, 3)) * np.nan
    for t in use_tiles:
        y, x = tilepos_yx[t]
        shift_heatmap[y, x] = nb.stitch.shifts[t]

    # plot the heatmaps
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["y shift", "x shift", "z shift"]
    for i in range(3):
        im = ax[i].imshow(shift_heatmap[:, :, i], cmap="bwr")
        # add text annotations of the tile number in the center of each tile
        for y, x in np.ndindex(n_rows, n_cols):
            tile_t = np.where((tilepos_yx == [y, x]).all(axis=1))[0]
            ax[i].text(x, y, tile_t, ha="center", va="center", color="black")
        ax[i].set_title(labels[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        fig.colorbar(im, ax=ax[i])

    plt.show()
