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

    for i, r in tqdm.tqdm(enumerate(nbp_basic.use_rounds), desc="Loading spot colours", unit="round"):
        image_r = preprocessing.load_icp_corrected_images(
            nbp_basic, nbp_file, nbp_extract, nbp_register, tile, r, nbp_basic.use_channels
        )
        image_r = (image_r + nbp_basic.tile_pixel_value_shift).astype(dtype)
        image_r = image_r.transpose((1, 2, 3, 0))
        colours[:, :, :, i] = image_r

    return colours.astype(dtype)
