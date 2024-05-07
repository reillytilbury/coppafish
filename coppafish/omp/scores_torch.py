import torch
import numpy as np


def score_coefficient_image(
    coefficient_image: torch.Tensor,
    spot: torch.Tensor,
    mean_spot: torch.Tensor,
    high_coefficient_bias: float,
    force_cpu: bool = True,
):
    """
    See omp/scores.py for a docstring description.
    """
    assert coefficient_image.dim() == 4
    assert spot.dim() == 3
    assert torch.isin(spot, torch.asarray([-1, 0, 1], device=coefficient_image.device)).all()
    assert spot.shape == mean_spot.shape
    assert torch.logical_and(mean_spot >= -1, mean_spot <= 1).all()
    assert high_coefficient_bias >= 0

    cpu = torch.device("cpu")
    run_on = cpu
    if not force_cpu and torch.cuda.is_available():
        run_on = torch.device("cuda")

    n_genes = coefficient_image.shape[3]

    coefficient_image = coefficient_image.to(device=run_on)
    spot = spot.to(device=run_on)
    mean_spot = mean_spot.to(device=run_on)

    spot_shape_kernel = torch.zeros_like(spot, dtype=mean_spot.dtype, device=run_on)
    spot_shape_kernel[spot == 1] = mean_spot[spot == 1]
    spot_shape_kernel /= spot_shape_kernel.sum()

    coefficient_image_function = coefficient_image.detach().clone()
    positive = coefficient_image > 0
    coefficient_image_function[~positive] = 0
    coefficient_image_function[positive] = coefficient_image_function[positive] / (
        coefficient_image_function[positive] + high_coefficient_bias
    )

    result = torch.zeros_like(
        coefficient_image_function, dtype=coefficient_image.dtype, device=coefficient_image.device
    )
    for g in range(n_genes):
        result[:, :, :, g] = torch.nn.functional.conv3d(
            coefficient_image_function[np.newaxis, np.newaxis, :, :, :, g],
            spot_shape_kernel[np.newaxis, np.newaxis],
            padding="same",
            bias=None,
        )[0, 0]
    return torch.clip(result, 0, 1).to(device=cpu, dtype=coefficient_image.dtype)


def omp_scores_float_to_int(scores: torch.Tensor) -> torch.Tensor:
    assert (0 <= scores).all() and (scores <= 1).all(), "scores should be between 0 and 1 inclusive"

    return torch.round(scores * np.iinfo(np.int16).max, decimals=0).to(torch.int16)


def omp_scores_int_to_float(scores: torch.Tensor) -> torch.Tensor:
    assert (0 <= scores).all() and (scores <= np.iinfo(np.int16).max).all()

    return (scores.float() / np.iinfo(np.int16).max).float()
