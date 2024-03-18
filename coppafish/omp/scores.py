import scipy
import numpy as np
import numpy.typing as npt

from .. import logging


def score_coefficient_image(
    coefs_image: np.ndarray,
    spot_shape: np.ndarray,
    spot_shape_mean: np.ndarray,
    high_coef_bias: float,
) -> npt.NDArray[np.float32]:
    """
    Computes OMP score(s) for the coefficient image. This is a weighted average of spot_shape_mean at spot_shape's == 1
    in a local area with their corresponding functioned coefficients. Effectively just a convolution.

    Args:
        coefs_image (`(im_y x im_x x im_z x n_genes) ndarray[float32]`): OMP coefficients in 3D shape. Any non-computed
            or out of bounds coefficients will be zero.
        spot_shape (`(size_y x size_x x size_z) ndarray[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
        spot_shape_mean (`(size_y x size_x x size_z) ndarray[float]`): OMP mean spot shape. This can range from -1 and
            1.
        high_coef_bias (float): specifies the constant used in the function applied to every coefficient. The function
            applied is `c / (c + high_coef_bias)` if c >= 0, 0 otherwise, where c is a coefficient value. This places
            higher scoring on larger coefficients.

    Returns:
        `(im_y x im_x x im_z x n_genes) ndarray[float]`: score for each coefficient pixel.
    """
    assert coefs_image.ndim == 4, "coefs_image must be four-dimensional"
    assert spot_shape.ndim == 3, "spot_shape must be three-dimensional"
    assert np.isin(spot_shape, [-1, 0, 1]).all(), "OMP spot shape should only contain -1, 0, and 1"
    assert spot_shape.shape == spot_shape_mean.shape, "spot_shape and spot_shape_mean should have the same shape"
    assert np.logical_and(-1 <= spot_shape_mean, spot_shape_mean <= 1).all(), "spot_shape_mean must range -1 to 1"
    assert high_coef_bias >= 0, "high_coef_bias cannot be negative"

    n_genes = coefs_image.shape[3]

    # Step 1: Retrieve the spot shape kernel. The kernel is zero where the spot shape is -1 or 0. The kernel is equal
    # to the spot shape mean where the spot shape is 1.
    spot_shape_kernel = np.zeros_like(spot_shape, dtype=np.float32)
    spot_shape_kernel[spot_shape == 1] = spot_shape_mean[spot_shape == 1]
    # Normalised like this s.t. all scores range from 0 to 1.
    spot_shape_kernel /= spot_shape_kernel.sum()
    n_shifts = (spot_shape == 1).sum()
    message = f"OMP gene scores are being computed with {n_shifts} local coefficients for each spot."
    if n_shifts < 25:
        message += f" Consider reducing the shape_sign_thresh in OMP config"
        logging.warn(message)
    else:
        logging.debug(message)

    # Step 2: Apply the non-linear function x / (x + high_coef_bias) to every positive coefficient element-wise, where
    # x is the coefficient. All negative coefficients are set to zero.
    coefs_image_function = coefs_image.copy()
    positive = coefs_image > 0
    coefs_image_function[~positive] = 0
    coefs_image_function[positive] = coefs_image_function[positive] / (coefs_image_function[positive] + high_coef_bias)

    # Step 3: 3D Convolve the functioned coefficients with the spot shape kernel
    result = np.zeros_like(coefs_image_function, dtype=np.float32)
    for g in range(n_genes):
        # This scipy convolve will automatically use zeroes when on the edge of the image
        result[:, :, :, g] = scipy.signal.convolve(coefs_image_function[:, :, :, g], spot_shape_kernel, mode="same")
    return np.clip(result, 0, 1, dtype=np.float32)


def omp_scores_float_to_int(scores: npt.NDArray[np.float_]) -> npt.NDArray[np.int16]:
    assert (0 <= scores).all() and (scores <= 1).all(), "scores should be between 0 and 1 inclusive"

    return np.round(scores * np.iinfo(np.int16).max, 0).astype(np.int16)


def omp_scores_int_to_float(scores: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    assert (0 <= scores).all() and (scores <= np.iinfo(np.int16).max).all()

    return (scores / np.iinfo(np.int16).max).astype(np.float32)
