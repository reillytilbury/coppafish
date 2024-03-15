import numpy as np
from scipy.sparse import csr_matrix
import numpy.typing as npt
from typing import Union

from .. import logging
from ..utils import morphology


def score_coefficient_image(
    coefs_image: Union[np.ndarray, csr_matrix],
    spot_shape: np.ndarray,
    spot_shape_mean: np.ndarray,
    high_coef_bias: float,
) -> npt.NDArray[np.float32]:
    """
    Computes OMP score(s) for the coefficient image. This is a weighted average of spot_shape_mean at spot_shape's == 1
    in a local area with their corresponding functioned coefficients. Effectively just a convolution.

    Args:
        coefs_image (`(im_y x im_x x im_z x n_genes) ndarray[float32]` or `csr_matrix`): OMP coefficients in 3D shape.
            Any non-computed or out of bounds coefficients will be zero.
        spot_shape (`(size_y x size_x x size_z) ndarray[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
        spot_shape_mean (`(size_y x size_x x size_z) ndarray[float]`): OMP mean spot shape. This can range from -1 and
            1.
        high_coef_bias (float): specifies the constant used in the function applied to every coefficient. The function
            applied is `c / (c + high_coef_bias)` if c >= 0, 0 otherwise, where c is a coefficient value. This places
            higher scoring on larger coefficients.

    Returns:
        `(im_y x im_x x im_z x n_genes) ndarray[float]`: score for each spot in spot_no if given, otherwise all
            n_pixels spot scores.
    """
    # NOTE: I re-wrote this function in pytorch and tested its speed. I found that it was just as slow, if not slower
    # since there are large ndarray -> tensor transformations that must be done. To speed this code up further, we
    # would require refactoring how the function works exactly or changing the inputs. But, currently its speed is not
    # horrendous (everything is entirely vectorised) and it puts OMP in a step in the right direction in terms of
    # scoring spots better.
    assert coefs_image.ndim == 4, "coefs_image must be four-dimensional"
    assert spot_shape.ndim == 3, "spot_shape must be three-dimensional"
    assert np.isin(spot_shape, [-1, 0, 1]).all(), "OMP spot shape should only contain -1, 0, and 1"
    assert spot_shape.shape == spot_shape_mean.shape, "spot_shape and spot_shape_mean should have the same shape"
    assert np.logical_and(-1 <= spot_shape_mean, spot_shape_mean <= 1).all(), "spot_shape_mean must range -1 to 1"
    assert high_coef_bias >= 0, "high_coef_bias cannot be negative"

    # Step 1: Get the neighbouring coefficients around each pixel where the spot shape is one (i.e. the coefficient is
    # expected to be positive).
    # (3, n_shifts)
    spot_shape_shifts_yxz = np.array(morphology.filter.get_shifts_from_kernel(spot_shape), dtype=int)
    im_y, im_x, im_z = coefs_image.shape[:3]
    n_shifts = spot_shape_shifts_yxz.shape[1]
    message = f"OMP gene scores are being computed with {n_shifts} local coefficients for each spot."
    if n_shifts < 25:
        message += f" Consider reducing the shape_sign_thresh in OMP config"
        logging.warn(message)
    else:
        logging.debug(message)
    mid_spot_shape_yxz = np.array(spot_shape.shape, dtype=int) // 2
    spot_shape_mean_consider = spot_shape_mean[tuple(mid_spot_shape_yxz[:, np.newaxis] + spot_shape_shifts_yxz)]
    # (3 x im_y x im_x x im_z)
    pixel_yxz_consider = np.array(
        np.meshgrid(np.arange(im_y), np.arange(im_x), np.arange(im_z)), dtype=np.int32
    ).swapaxes(1, 2)
    # (3 x im_y x im_x x im_z x n_shifts) all coordinate positions to consider for each coefs_image
    pixel_yxz_consider = pixel_yxz_consider[..., np.newaxis].repeat(n_shifts, axis=4)
    pixel_yxz_consider += spot_shape_shifts_yxz[:, np.newaxis, np.newaxis, np.newaxis]
    # Pad coefs_image with zeros for pixels on the outer edges of the image
    pad_widths = tuple([(0, spot_shape_shifts_yxz[i].max()) for i in range(3)]) + ((0, 0),)
    coefs_image_padded = np.pad(coefs_image, pad_widths, mode="constant", constant_values=0)
    # (im_y x im_x x im_z x n_shifts x n_genes)
    coefs_image_consider = coefs_image_padded[tuple(pixel_yxz_consider)]
    del pixel_yxz_consider, coefs_image_padded

    # Step 2: Since coefficients can range from -infinity to infinity, they are functioned element-wise to give values
    # from 0 to 1 using the formula x / (x + high_coef_bias) if x >= 0, 0 otherwise, where x is a coefficient.
    positive = coefs_image_consider > 0
    coefs_image_consider[~positive] = 0
    coefs_image_consider[positive] = coefs_image_consider[positive] / (coefs_image_consider[positive] + high_coef_bias)

    # Step 3: The functioned coefficients are then weight-averaged with the spot shape mean and divided such that the
    # scores range from 0 to 1.
    coefs_image_consider *= spot_shape_mean_consider[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    return coefs_image_consider.sum(axis=3) / spot_shape_mean_consider.sum()


def omp_scores_float_to_int(scores: npt.NDArray[np.float_]) -> npt.NDArray[np.int16]:
    assert (0 <= scores).all() and (scores <= 1).all(), "scores should be between 0 and 1 inclusive"

    return np.round(scores * np.iinfo(np.int16).max, 0).astype(np.int16)


def omp_scores_int_to_float(scores: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    assert (0 <= scores).all() and (scores <= np.iinfo(np.int16).max).all()

    return (scores / np.iinfo(np.int16).max).astype(np.float32)
