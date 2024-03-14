import numpy as np
from scipy.sparse import csr_matrix
import numpy.typing as npt
from typing import Union

from .. import logging
from ..utils import morphology


def score_omp_spots(
    spot_shape: np.ndarray,
    spot_shape_mean: np.ndarray,
    pixel_yxz: np.ndarray,
    pixel_coefs: Union[np.ndarray, csr_matrix],
    sigmoid_weight: float,
    spot_no: np.ndarray = None,
) -> npt.NDArray[np.float32]:
    """
    Computes score(s) for omp gene reads at `pixel_yxz[spot_no]` positions.

    Args:
        spot_shape (`(size_y x size_x x size_z) ndarray[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
        spot_shape_mean (`(size_y x size_x x size_z) ndarray[float]`): OMP mean spot shape. This can range from -1 and
            1.
        pixel_yxz (`(n_pixels x 3) ndarray[int]`): all local spot positions.
        pixel_coefs (`(n_pixels x n_genes) ndarray[float]` or csr_matrix): the gene weight assigned by OMP for each
            pixel and gene, most will be zero so it is often kept as a `scipy` csr matrix. Pixel at position
            `pixel_yxz[p]` has coefficient of `pixel_coefs[p]`.
        sigmoid_weight (float): specifies the sigmoid constant. A value of 0 gives equal weighting to all coefficients,
            whereas a value > 0 gives a stronger weighting to large coefficients.
        spot_no (`(n_pixels_consider) ndarray[int]`): Which spots to get score for. If `None`, all scores will be found.

    Returns:
        `(n_pixels_consider x n_genes) ndarray[float]`: score for each spot in spot_no if given, otherwise all n_pixels
            spot scores.
    """
    # NOTE: I re-wrote this function in pytorch and tested its speed. I found that it was just as slow, if not slower
    # since there are large ndarray -> tensor transformations that must be done. To speed this code up further, we
    # would require refactoring how the function works exactly or changing the inputs. But, currently its speed is not
    # horrendous (everything is entirely vectorised) and it puts OMP in a step in the right direction in terms of
    # scoring spots better.
    assert spot_shape.ndim == 3, "spot_shape must be three-dimensional"
    assert np.isin(spot_shape, [-1, 0, 1]).all(), "OMP spot shape should only contain -1, 0, and 1"
    assert spot_shape.shape == spot_shape_mean.shape, "spot_shape and spot_shape_mean should have the same shape"
    assert np.logical_and(-1 <= spot_shape_mean, spot_shape_mean <= 1).all(), "spot_shape_mean must range -1 to 1"
    assert sigmoid_weight >= 0, "sigmoid weighting cannot be negative"

    # Step 1: Gather the required coefficients from pixel_coefs. These are the coefficients around each
    # pixel_yxz[spot_no] position where the spot_shape is 1 (i.e. confidently expecting a positive coefficient there).
    # If the coefficient was not computed by OMP, then it is set to zero.
    spot_shape_shifts = np.array(morphology.filter.get_shifts_from_kernel(spot_shape), dtype=int)
    n_shifts = spot_shape_shifts.shape[1]
    message = f"OMP gene scores are being computed with {n_shifts} local coefficients for each spot."
    if n_shifts < 25:
        message += f" Consider reducing the shape_sign_thresh in OMP config"
        logging.warn(message)
    else:
        logging.debug(message)
    n_genes = pixel_coefs.shape[1]
    mid_spot_shape_yxz = np.array(spot_shape.shape, dtype=int) // 2
    spot_shape_mean_consider = spot_shape_mean[tuple(mid_spot_shape_yxz[:, np.newaxis] + spot_shape_shifts)]
    if spot_no is None:
        pixel_yxz_consider = pixel_yxz
    else:
        pixel_yxz_consider = pixel_yxz[spot_no]
    n_pixels_consider = pixel_yxz_consider.shape[0]
    # (3 x n_shifts x n_pixels_consider) positions to find coefficients for.
    pixel_yxz_consider = np.repeat(pixel_yxz_consider.T[:, np.newaxis], n_shifts, axis=1)
    pixel_yxz_consider += spot_shape_shifts[:, :, np.newaxis]
    # (3 x n_shifts * n_pixels_consider)
    pixel_yxz_consider = pixel_yxz_consider.reshape((3, -1))
    # Learn what pixel_yxz_consider are found in pixel_yxz/pixel_coefs so pixel_coefs_consider can be populated.
    # Build up a "coefficient image" based on the given pixel positions in pixel_yxz, with a zero padding in case of
    # spots on the edge of the image.
    coef_image_shape = tuple(np.abs(pixel_yxz).max(0) + np.abs(spot_shape_shifts).max(1) + 1) + (n_genes,)
    coef_image = np.zeros(coef_image_shape, dtype=np.float32)
    if isinstance(pixel_coefs, csr_matrix):
        coef_image[tuple(pixel_yxz.T)] = pixel_coefs.toarray().astype(np.float32)
    else:
        coef_image[tuple(pixel_yxz.T)] = pixel_coefs.astype(np.float32)
    pixel_coefs_consider = coef_image[tuple(pixel_yxz_consider)]
    del coef_image

    # Step 2: Since coefficients can range from 0 to infinity, they are sigmoided individually to give values from 0 to
    # 1 using the formula 1 / (sigmoid_weight * e^{- x} + 1) where x is a coefficient.
    pixel_coefs_consider = 1 / (1 + sigmoid_weight * np.exp(-pixel_coefs_consider))
    pixel_coefs_consider = pixel_coefs_consider.reshape((n_shifts, n_pixels_consider, n_genes))

    # Step 3: The sigmoided coefficients are then weight-averaged with the spot shape mean and divided such that the
    # scores range from 0 to 1.
    pixel_coefs_consider *= spot_shape_mean_consider[:, np.newaxis, np.newaxis]
    return pixel_coefs_consider.sum(axis=0) / spot_shape_mean_consider.sum()


def omp_scores_float_to_int(scores: npt.NDArray[np.float_]) -> npt.NDArray[np.int16]:
    assert (0 <= scores).all() and (scores <= 1).all(), "scores should be between 0 and 1 inclusive"

    return np.round(scores * np.iinfo(np.int16).max, 0).astype(np.int16)


def omp_scores_int_to_float(scores: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    assert (0 <= scores).all() and (scores <= np.iinfo(np.int16).max).all()

    return (scores / np.iinfo(np.int16).max).astype(np.float32)
