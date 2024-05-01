import scipy
import numpy as np
import numpy.typing as npt

from .. import log


def score_coefficient_image(
    coefficient_image: np.ndarray,
    spot: np.ndarray,
    mean_spot: np.ndarray,
    high_coefficient_bias: float,
    force_cpu: bool = True,
) -> npt.NDArray[np.float32]:
    """
    Computes OMP score(s) for the coefficient image. This is a weighted average of spot_shape_mean at spot_shape's == 1
    in a local area with their corresponding functioned coefficients. Effectively just a convolution.

    Args:
        coefs_image (`(im_y x im_x x im_z x n_genes) ndarray[float32]`): OMP coefficients in 3D shape. Any non-computed
            or out of bounds coefficients will be zero.
        spot (`(size_y x size_x x size_z) ndarray[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
        mean_spot (`(size_y x size_x x size_z) ndarray[float]`): OMP mean spot shape. This can range from -1 and
            1.
        high_coef_bias (float): specifies the constant used in the function applied to every coefficient. The function
            applied is `c / (c + high_coef_bias)` if c >= 0, 0 otherwise, where c is a coefficient value. This places
            higher scoring on larger coefficients.

    Returns:
        `(im_y x im_x x im_z x n_genes) ndarray[float]`: score for each coefficient pixel.
    """
    assert coefficient_image.ndim == 4, "coefs_image must be four-dimensional"
    assert spot.ndim == 3, "spot must be three-dimensional"
    assert np.isin(spot, [-1, 0, 1]).all(), "spot can only contain -1, 0, and 1"
    assert spot.shape == mean_spot.shape, "spot and mean_spot must have the same shape"
    assert np.logical_and(-1 <= mean_spot, mean_spot <= 1).all(), "mean_spot must range -1 to 1"
    assert high_coefficient_bias >= 0, "high_coef_bias cannot be negative"

    n_genes = coefficient_image.shape[3]

    # Step 1: Retrieve the spot shape kernel. The kernel is zero where the spot shape is -1 or 0. The kernel is equal
    # to the spot shape mean where the spot shape is 1.
    spot_shape_kernel = np.zeros_like(spot, dtype=coefficient_image.dtype)
    spot_shape_kernel[spot == 1] = mean_spot[spot == 1]
    # Normalised like this s.t. all scores range from 0 to 1.
    spot_shape_kernel /= spot_shape_kernel.sum()
    n_shifts = (spot == 1).sum()
    message = f"OMP gene scores are being computed with {n_shifts} local coefficients for each spot."
    if n_shifts < 20:
        message += f" You may need to reduce shape_sign_thresh in OMP config"
        if n_shifts == 0:
            raise ValueError(message)
        log.warn(message)
    else:
        log.debug(message)

    # Step 2: Apply the non-linear function x / (x + high_coef_bias) to every positive coefficient element-wise, where
    # x is the coefficient. All negative coefficients are set to zero.
    coefs_image_function = coefficient_image.copy()
    positive = coefficient_image > 0
    coefs_image_function[~positive] = 0
    coefs_image_function[positive] = coefs_image_function[positive] / (
        coefs_image_function[positive] + high_coefficient_bias
    )

    # Step 3: 3D Convolve the functioned coefficients with the spot shape kernel
    result = np.zeros_like(coefs_image_function, dtype=coefficient_image.dtype)
    for g in range(n_genes):
        # This scipy convolve will automatically use zeroes when on the edge of the image
        result[:, :, :, g] = scipy.signal.convolve(coefs_image_function[:, :, :, g], spot_shape_kernel, mode="same")
    return np.clip(result, 0, 1, dtype=coefficient_image.dtype)


def omp_scores_float_to_int(scores: npt.NDArray[np.float_]) -> npt.NDArray[np.int16]:
    assert (0 <= scores).all() and (scores <= 1).all(), "scores should be between 0 and 1 inclusive"

    return np.round(scores * np.iinfo(np.int16).max, 0).astype(np.int16)


def omp_scores_int_to_float(scores: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    assert (0 <= scores).all() and (scores <= np.iinfo(np.int16).max).all()

    return (scores / np.iinfo(np.int16).max).astype(np.float32)
