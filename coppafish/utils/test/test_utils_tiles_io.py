import os
import numpy as np

from coppafish import NotebookPage
from coppafish.utils import tiles_io


def test_save_load_image():
    file_type = ".zarr"
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "unit_test_dir")

    if not os.path.isdir(directory):
        os.mkdir(directory)
    rng = np.random.RandomState(0)
    array_1_shape = (3, 3, 4)
    array_1 = rng.rand(*array_1_shape).astype(dtype=np.float16)
    array_1_path = os.path.join(directory, "array_1.zarr")

    tiles_io._save_image(array_1, array_1_path)

    array_1_returned = tiles_io._load_image(array_1_path)
    assert np.allclose(array_1_returned, array_1)


# TODO: get_npy_tile_ind unit tests.
