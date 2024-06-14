import torch
import numpy as np
from typing import Tuple

from ..utils.morphology import filter


def compute_mean_spot_from(
    image: torch.Tensor,
    spot_positions_yxz: torch.Tensor,
    spot_shape: Tuple[int, int, int],
    force_cpu: bool = True,
) -> torch.Tensor:
    """
    Compute the mean spot from the given positions on the given image in a cuboid local region around each spot.

    Args:
        image (`(im_y x im_x x im_z) tensor`): image. Any out of bounds retrievals around spots are set to zero.
        spot_positions_yxz (`(n_spots x 3) tensor`): every spot position to use to compute the spot.
        spot_shape (`tuple of three ints`): spot size in y, x, and z respectively.
        mean_sign_threshold (float): any mean spot shape value above this threshold is set to one in the spot shape.
        force_cpu (bool): only use the CPU to run computations.

    Returns:
        (`spot_shape tensor`) mean_spot: the mean of the signs of the coefficient.
    """
    assert type(image) is torch.Tensor
    assert type(spot_positions_yxz) is torch.Tensor
    assert image.dim() == 3
    assert spot_positions_yxz.dim() == 2
    assert spot_positions_yxz.shape[0] > 0, "require at least one spot position to compute the spot shape from"
    assert len(spot_shape) == 3, "spot_shape must be a tuple with 3 integer numbers"
    assert (torch.asarray(spot_shape) % 2 != 0).all(), "spot_shape must be all odd numbers"

    cpu = torch.device("cpu")
    run_on = cpu
    if not force_cpu and torch.cuda.is_available():
        run_on = torch.device("cuda")

    spot_positions_yxz = spot_positions_yxz.to(run_on)
    # Pad the image with zeros on one edge for every dimension so out of bounds retrievals are all zeros.
    image_padded = torch.nn.functional.pad(image, (0, spot_shape[2], 0, spot_shape[1], 0, spot_shape[0])).to(run_on)

    n_spots = spot_positions_yxz.shape[0]
    mean_spot = torch.zeros(spot_shape, dtype=torch.float32, device=run_on)

    # (3, n_shifts)
    spot_shifts = torch.asarray(
        np.array(filter.get_shifts_from_kernel(np.ones(spot_shape))), dtype=torch.int16, device=run_on
    )
    n_shifts = spot_shifts.shape[1]
    # (3, n_shifts)
    spot_shift_positions = spot_shifts.int() + (torch.asarray(spot_shape, dtype=int, device=run_on) // 2)[:, np.newaxis]
    # (3, 1, n_spots)
    spot_positions_yxz = spot_positions_yxz.T[:, np.newaxis].repeat_interleave(n_shifts, dim=1)
    # (3, n_shifts, n_spots)
    spot_positions_yxz += spot_shifts[:, :, np.newaxis]
    # (3, n_shifts * n_spots)
    spot_positions_yxz = spot_positions_yxz.reshape((3, -1))

    # (n_shifts * n_spots)
    spot_image_values = torch.sign(image_padded[tuple(spot_positions_yxz)]).float()

    mean_spot[tuple(spot_shift_positions)] = spot_image_values.reshape((n_shifts, n_spots)).mean(dim=1)

    return mean_spot.to(cpu)


def count_edge_ones(
    spot: torch.Tensor,
) -> int:
    """
    Counts the number of ones on the x and y edges for all z planes.

    Args:
        spot (`(size_y x size_x x size_z) tensor[int]`): OMP spot shape. It is a made up of only zeros and ones.
            Ones indicate where the spot coefficient is likely to be positive.
    """
    assert type(spot) is torch.Tensor
    assert spot.dim() == 3
    assert torch.isin(spot, torch.asarray([0, 1], device=spot.device)).all()

    count = 0
    for z in range(spot.shape[2]):
        count += spot[:, :, z].sum() - spot[1 : spot.shape[0] - 1, 1 : spot.shape[1] - 1, z].sum()
    return int(count)
