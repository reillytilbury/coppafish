import tqdm
import numpy as np
from typing_extensions import assert_type
import numpy.typing as npt

from ..register import preprocessing
from ..setup.notebook import NotebookPage


def load_spot_colours(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_register: NotebookPage,
    nbp_register_debug: NotebookPage,
    tile: int,
    dtype: np.dtype = np.uint16,
) -> npt.NDArray:
    """
    Load the full registered image for every sequencing round/channel for the given tile. No post-processing is
    applied, including tile_pixel_value_shift subtraction, background subtraction, and colour normalisation.

    Returns:
        (`(im_y x im_x x im_z x n_rounds_use x n_channels_use) ndarray`) spot_colours: tile loaded spot colours.
    """
    assert_type(nbp_basic, NotebookPage)
    assert_type(nbp_file, NotebookPage)
    assert_type(nbp_extract, NotebookPage)
    assert_type(nbp_register, NotebookPage)
    assert_type(nbp_register_debug, NotebookPage)
    assert_type(tile, int)
    assert_type(dtype, np.dtype)

    image_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    colours = np.zeros(image_shape + (len(nbp_basic.use_rounds), len(nbp_basic.use_channels)), dtype=dtype)

    with tqdm.tqdm(
        total=len(nbp_basic.use_channels) * len(nbp_basic.use_rounds), desc=f"Loading spot colours, {tile=}"
    ) as pbar:
        for i, r in enumerate(nbp_basic.use_rounds):
            for j, c in enumerate(nbp_basic.use_channels):
                image_rc = preprocessing.load_transformed_image(
                    nbp_basic,
                    nbp_file,
                    nbp_extract,
                    nbp_register,
                    nbp_register_debug,
                    tile,
                    r,
                    c,
                    reg_type="flow_icp",
                )
                # In the preprocessing function, the images are shifted by -15_000 to centre the zero in the correct
                # place. We are undoing this here so the images can be stored as uint's in memory.
                image_rc = (image_rc + nbp_basic.tile_pixel_value_shift).astype(dtype)
                colours[:, :, :, i, j] = image_rc
                del image_rc
                pbar.update()

    return colours.astype(dtype)
