from typing import Tuple

import numpy as np


def get_pixel_length(length_microns: float, pixel_size: float) -> int:
    """
    Converts a length in units of microns into a length in units of pixels

    Args:
        length_microns: Length in units of microns (microns)
        pixel_size: Size of a pixel in microns (microns/pixels)

    Returns:
        Desired length in units of pixels (pixels)

    """
    return int(round(length_microns / pixel_size))


def strip_hack(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds all columns in image where each row is identical and then sets
    this column to the nearest normal column. Basically 'repeat padding'.

    Args:
        image: ```float [n_y x n_x (x n_z)]```
            Image from nd2 file, before filtering (can be after focus stacking) and if 3d, last index must be z.

    Returns:
        - ```image``` - ```float [n_y x n_x (x n_z)]```
            Input array with change_columns set to nearest
        - ```change_columns``` - ```int [n_changed_columns]```
            Indicates which columns have been changed.
    """
    # all rows identical if standard deviation is 0
    if np.ndim(image) == 3:
        # assume each z-plane of 3d image has same bad columns
        # seems to always be the case for our data
        change_columns = np.where(np.std(image[:, :, 0], 0) == 0)[0]
    else:
        change_columns = np.where(np.std(image, 0) == 0)[0]
    good_columns = np.setdiff1d(np.arange(np.shape(image)[1]), change_columns)
    for col in change_columns:
        nearest_good_col = good_columns[np.argmin(np.abs(good_columns - col))]
        image[:, col] = image[:, nearest_good_col]
    return image, change_columns
