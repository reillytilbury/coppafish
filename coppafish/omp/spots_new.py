import numpy as np
from typing_extensions import assert_type
import numpy.typing as npt
from typing import Tuple

from ..utils.morphology import filter


def compute_mean_spot_from(
    image: npt.NDArray[np.float32],
    spot_positions_yxz: npt.NDArray[np.int16],
    spot_shape: Tuple[int, int, int],
) -> npt.NDArray[np.float32]:
    """
    Compute the mean spot from the given positions on the given image in a cuboid local region around each spot.

    Args:
        image (`(im_y x im_x x im_z) ndarray`): image. Any out of bounds retrievals around spots are set to zero.
        spot_positions_yxz (`(n_spots x 3) ndarray`): every spot position to use to compute the spot.
        spot_shape (`tuple of three ints`): spot size in y, x, and z respectively.
        mean_sign_threshold (float): any mean spot shape value above this threshold is set to one in the spot shape.

    Returns:
        (`spot_shape ndarray`) mean_spot: the mean of the signs of the coefficient.
    """
    assert_type(image, np.ndarray)
    assert image.ndim == 3
    assert_type(spot_positions_yxz, np.ndarray)
    assert spot_positions_yxz.ndim == 2
    assert spot_positions_yxz.shape[0] > 0, "require at least one spot position to compute the spot shape from"
    assert len(spot_shape) == 3, "spot_shape must be a tuple with 3 integer numbers"
    assert (np.array(spot_shape) % 2 != 0).all(), "spot_shape must be all odd numbers"

    n_spots = spot_positions_yxz.shape[0]

    # Pad the image with zeros on one edge for every dimension so out of bounds retrievals are all zeros.
    image_padded = np.pad(image, ((0, spot_shape[0]), (0, spot_shape[1]), (0, spot_shape[2])))

    mean_spot = np.zeros(spot_shape, dtype=np.float32)

    # (3, n_shifts)
    spot_shifts = np.array(filter.get_shifts_from_kernel(np.ones(spot_shape)), dtype=np.int16)
    n_shifts = spot_shifts.shape[1]
    # (3, n_shifts)
    spot_shift_positions = spot_shifts + (np.array(spot_shape, np.int16) // 2)[:, np.newaxis]
    # (3, 1, n_spots)
    spot_positions_yxz = spot_positions_yxz.T[:, np.newaxis].repeat(n_shifts, axis=1)
    # (3, n_shifts, n_spots)
    spot_positions_yxz += spot_shifts[:, :, np.newaxis]
    # (3, n_shifts * n_spots)
    spot_positions_yxz = spot_positions_yxz.reshape((3, -1))

    # (n_shifts * n_spots)
    spot_image_values = np.sign(image_padded[tuple(spot_positions_yxz)], dtype=np.float32)

    mean_spot[tuple(spot_shift_positions)] = spot_image_values.reshape((n_shifts, n_spots)).mean(axis=1)

    return mean_spot
