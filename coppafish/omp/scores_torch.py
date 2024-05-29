import math as maths

import numpy as np
import torch
from typing_extensions import assert_type


def score_coefficient_image(
    coefficient_image: torch.Tensor,
    points: torch.Tensor,
    spot: torch.Tensor,
    mean_spot: torch.Tensor,
    high_coefficient_bias: float,
    force_cpu: bool = True,
):
    """
    Computes OMP score(s) for the coefficient image. This is a weighted average of spot_shape_mean at spot_shape's == 1
    in a local area with their corresponding functioned coefficients. Effectively just a convolution.

    Args:
        coefs_image (`(im_y x im_x x im_z) tensor[float32]`): OMP coefficients in 3D shape. Any non-computed
            or out of bounds coefficients will be zero.
        points (`(n_points x 3) tensor[int]`): points to be scored.
        spot (`(size_y x size_x x size_z) tensor[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
        mean_spot (`(size_y x size_x x size_z) tensor[float32]`): OMP mean spot shape. This can range from -1 and
            1.
        high_coef_bias (float): specifies the constant used in the function applied to every coefficient. The function
            applied is `c / (c + high_coef_bias)` if c >= 0, 0 otherwise, where c is a coefficient value. This places
            higher scoring on larger coefficients.

    Returns:
        `(n_points) tensor[float]`: score for each coefficient pixel.
    """
    assert_type(coefficient_image, torch.tensor)
    assert_type(points, torch.tensor)
    assert_type(spot, torch.tensor)
    assert_type(mean_spot, torch.tensor)
    assert coefficient_image.dim() == 3
    assert points.dim() == 2
    assert points.shape[0] > 0
    assert points.shape[1] == 3
    assert spot.dim() == 3
    assert torch.isin(spot, torch.asarray([0, 1], device=coefficient_image.device)).all()
    assert spot.shape == mean_spot.shape
    assert torch.logical_and(mean_spot >= -1, mean_spot <= 1).all()
    assert high_coefficient_bias >= 0

    cpu = torch.device("cpu")
    run_on = cpu
    if not force_cpu and torch.cuda.is_available():
        run_on = torch.device("cuda")

    coefficient_image = coefficient_image.to(device=run_on)
    spot = spot.to(device=run_on)
    mean_spot = mean_spot.to(device=run_on)

    spot_shape_kernel = torch.zeros_like(spot, dtype=mean_spot.dtype, device=run_on)
    spot_shape_kernel[spot == 1] = mean_spot[spot == 1]
    spot_shape_kernel /= spot_shape_kernel.sum()

    coefficient_image_function = coefficient_image.detach().clone()

    # Crop the image to bound around points for faster computation.
    point_min, point_max = points.min(dim=0)[0], points.max(dim=0)[0]
    kernel_radius = torch.asarray([maths.ceil(spot_shape_kernel.shape[i] / 2) for i in range(3)], dtype=int)
    point_min -= kernel_radius
    point_max += kernel_radius
    point_min = torch.clamp(point_min, min=0)
    point_max = torch.clamp(point_max, max=torch.asarray(coefficient_image_function.shape))
    coefficient_image_function = coefficient_image_function[
        point_min[0] : point_max[0], point_min[1] : point_max[1], point_min[2] : point_max[2]
    ]

    positive = coefficient_image_function > 0
    coefficient_image_function[~positive] = 0
    coefficient_image_function[positive] /= coefficient_image_function[positive] + high_coefficient_bias

    results = torch.nn.functional.conv3d(
        coefficient_image_function[np.newaxis, np.newaxis],
        spot_shape_kernel[np.newaxis, np.newaxis],
        padding="same",
        bias=None,
    )[0, 0]
    results = results[tuple((points - point_min[np.newaxis]).T)]

    return torch.clip(results, 0, 1).to(device=cpu, dtype=coefficient_image.dtype)
