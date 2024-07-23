from typing import Any, Tuple, Union

import numpy as np
import scipy
import torch

from ..utils.morphology import filter


def is_duplicate_spot(yxz_global_positions: torch.Tensor, tile_number: int, tile_centres: torch.Tensor) -> torch.Tensor:
    """
    Checks what spot positions are duplicates. A duplicate is defined as any spot that is closer to a different tile
    origin than the one it is assigned to.

    Args:
        - yxz_global_positions (`(n_points x 3) tensor[int]`): y, x, and z global positions for each spot.
        - tile_number (int): the tile index for all spot positions.
        - tile_centres (`(n_tiles x 3) tensor[float]`): each tile's centre in global coordinates.

    Returns:
        (`(n_points) tensor[bool]`): true for each duplicate spot.
    """
    assert type(yxz_global_positions) is torch.Tensor
    n_points = yxz_global_positions.shape[0]
    assert n_points > 0, "Require at least one spot"
    assert yxz_global_positions.shape[1] == 3
    assert type(tile_number) is int
    assert type(tile_centres) is torch.Tensor
    assert tile_centres.shape[1] == 3
    assert tile_number >= 0 and tile_number < tile_centres.shape[0]

    kdtree = scipy.spatial.KDTree(tile_centres.numpy())
    # Find the nearest tile origin for each spot position.
    # If this is not the tile number assigned to the spot, it is a duplicate.
    closest_tile_numbers = kdtree.query(yxz_global_positions.numpy(), k=1)[1]
    closest_tile_numbers = np.array(closest_tile_numbers)
    closest_tile_numbers = torch.asarray(closest_tile_numbers)
    is_duplicate = closest_tile_numbers != tile_number

    return is_duplicate


def is_true_isolated(
    yxz_positions: torch.Tensor, distance_threshold_yx: Union[float, int], distance_threshold_z: Union[float, int]
) -> torch.Tensor:
    """
    Checks what point positions are truly isolated. A point is truly isolated if the closest other point position is
    further than the given distance thresholds.

    Args:
        - yxz_positions (`(n_points x 3) tensor[int]`): y, x, and z positions for each point.
        - distance_threshold_yx (float): any positions within this distance threshold along x or y are not truly
            isolated.
        - distance_threshold_z (float): any positions within this distance threshold along z are not truly isolated.

    Returns:
        (`(n_points) tensor[bool]`): true for each point considered truly isolated.
    """
    assert type(yxz_positions) is torch.Tensor
    assert yxz_positions.dim() == 2
    assert yxz_positions.shape[0] > 0
    assert yxz_positions.shape[1] == 3
    assert type(distance_threshold_yx) is float or type(distance_threshold_yx) is int
    assert type(distance_threshold_z) is float or type(distance_threshold_z) is int

    yxz_norm = yxz_positions.numpy()
    yxz_norm = yxz_norm.astype(np.float32)
    yxz_norm[:, 2] *= distance_threshold_yx / distance_threshold_z
    kdtree = scipy.spatial.KDTree(yxz_norm)
    close_pairs = kdtree.query_pairs(r=distance_threshold_yx, output_type="ndarray")
    assert close_pairs.shape[1] == 2
    close_pairs = close_pairs.ravel()
    close_pairs = np.unique(close_pairs)
    true_isolate = np.ones(yxz_norm.shape[0], dtype=bool)
    true_isolate[close_pairs] = False

    return true_isolate


def compute_mean_spot(
    coefficients: Any,
    spot_positions_yxz: torch.Tensor,
    spot_positions_gene_no: torch.Tensor,
    tile_shape: Tuple[int, int, int],
    spot_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Compute the mean spot from the given positions on the coefficient images in a cuboid local region centred around each
    given spot position. The mean spot is the mean of the image signs in the cuboid region.

    Args:
        coefficients (`(n_pixels x n_genes) scipy.sparse.csr_matrix`): coefficient images. Any out of bounds
            retrievals around spots are set to zero.
        spot_positions_yxz (`(n_spots x 3) tensor`): every spot position to use to compute the spot. If n_spots is 0,
            a mean spot of zeros is returned.
        spot_positions_gene_no (`(n_spots) tensor[int]`): every spot position's gene number.
        tile_shape (tuple of three ints): the tile's shape in y, x, and z.
        spot_shape (tuple of three ints): spot size in y, x, and z respectively. This is the size of the cuboids
            around each spot position. This must be an odd number in each dimension so the spot position can be centred.

    Returns:
        (`spot_shape tensor[float32]`) mean_spot: the mean of the signs of the coefficient.
    """
    assert type(coefficients) is scipy.sparse.csr_matrix
    assert type(spot_positions_yxz) is torch.Tensor
    assert spot_positions_yxz.dim() == 2
    n_spots = int(spot_positions_yxz.shape[0])
    assert n_spots > 0
    assert spot_positions_yxz.shape[1] == 3
    assert type(spot_positions_gene_no) is torch.Tensor
    assert spot_positions_gene_no.dim() == 1
    assert spot_positions_yxz.shape[0] == spot_positions_gene_no.shape[0]
    assert type(tile_shape) is tuple
    assert len(tile_shape) == 3
    assert coefficients.shape[0] == np.prod(tile_shape)
    assert type(spot_shape) is tuple
    assert len(spot_shape) == 3
    assert all([type(spot_shape[i]) is int for i in range(3)])
    assert (torch.asarray(spot_shape) % 2 != 0).all(), "spot_shape must be only odd numbers"

    spot_shifts = np.array(filter.get_shifts_from_kernel(np.ones(spot_shape)))
    spot_shifts = torch.asarray(spot_shifts).int()
    n_shifts = spot_shifts.size(1)
    # (3, n_shifts)
    spot_shift_positions = spot_shifts + (torch.asarray(spot_shape, dtype=int) // 2)[:, np.newaxis]

    spots = torch.zeros((0, n_shifts)).float()

    for g in spot_positions_gene_no.unique():
        g_coef_image = torch.asarray(coefficients[:, [g]].toarray()).reshape(tile_shape).float()
        # Pad the coefficient image for out of bound cases.
        g_coef_image = torch.nn.functional.pad(g_coef_image, (0, spot_shape[2], 0, spot_shape[1], 0, spot_shape[0]))
        g_yxz = spot_positions_yxz[spot_positions_gene_no == g].int()
        # (3, n_shifts, n_spots)
        g_spot_positions_yxz = g_yxz.T[:, np.newaxis].repeat_interleave(n_shifts, dim=1)
        g_spot_positions_yxz += spot_shifts[:, :, np.newaxis]
        # (n_shifts, n_spots)
        g_spots = g_coef_image[tuple(g_spot_positions_yxz)].float()
        # (g_n_spots, n_shifts)
        g_spots = g_spots.T

        spots = torch.cat((spots, g_spots), dim=0)

    assert spots.shape == (n_spots, n_shifts)
    mean_spot = torch.zeros(spot_shape).float()
    mean_spot[tuple(spot_shift_positions)] = spots.sign().mean(dim=0)

    return mean_spot


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
