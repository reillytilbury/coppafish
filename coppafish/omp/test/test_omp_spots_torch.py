import numpy as np
import scipy
import torch

from coppafish.omp.spots_torch import is_true_isolated, compute_mean_spot


def test_is_true_isolated() -> None:
    yxz_positions = torch.zeros((4, 3)).int()
    yxz_positions[0, 2] = 100
    yxz_positions[1, 0] = 3
    yxz_positions[2, 1] = 5
    distance_threshold_yx = 10
    distance_threshold_z = 2.0
    is_isolated = is_true_isolated(yxz_positions, distance_threshold_yx, distance_threshold_z)
    assert is_isolated[0]
    assert not is_isolated[1]
    assert not is_isolated[2]


def test_compute_spots_from() -> None:
    rng = np.random.RandomState(0)
    im_y, im_x, im_z = 10, 11, 12
    tile_shape = (im_y, im_x, im_z)
    image = scipy.sparse.csr_matrix(rng.rand(im_y, im_x, im_z).astype(np.float32).reshape((-1, 1)))
    image_numpy = image[:, [0]].toarray().reshape(tile_shape)
    # The most simple case possible.
    spot_positions_yxz = torch.zeros((1, 3)).int()
    spot_positions_yxz[0, 0] = 5
    spot_positions_yxz[0, 1] = 5
    spot_positions_yxz[0, 2] = 6
    spot_positions_gene_no = torch.zeros(1).int()
    spot_shape = (1, 1, 1)
    mean_spot = compute_mean_spot(image, spot_positions_yxz, spot_positions_gene_no, tile_shape, spot_shape)
    assert np.allclose(mean_spot, np.sign(image_numpy[5, 5, 6]))

    # Two spot positions
    spot_positions_yxz = torch.zeros((2, 3)).int()
    spot_positions_yxz[0, 0] = 5
    spot_positions_yxz[0, 1] = 5
    spot_positions_yxz[0, 2] = 6
    spot_positions_yxz[1, 0] = 4
    spot_positions_yxz[1, 1] = 5
    spot_positions_yxz[1, 2] = 7
    spot_positions_gene_no = torch.zeros(2).int()
    mean_spot = compute_mean_spot(image, spot_positions_yxz, spot_positions_gene_no, tile_shape, spot_shape)
    signs = torch.sign(torch.asarray([image_numpy[5, 5, 6], image_numpy[4, 5, 7]]))
    assert torch.allclose(mean_spot, torch.mean(signs))

    # Two large spot shapes with out of bounds cases.
    spot_shape = (15, 11, 3)
    mean_spot = compute_mean_spot(image, spot_positions_yxz, spot_positions_gene_no, tile_shape, spot_shape)

    spot_shape = (3, 1, 1)
    spot_positions_yxz = torch.zeros((2, 3)).int()
    spot_positions_yxz[0, 0] = 3
    spot_positions_yxz[0, 1] = 4
    spot_positions_yxz[0, 2] = 1
    spot_positions_yxz[1, 0] = 6
    spot_positions_yxz[1, 1] = 6
    spot_positions_yxz[1, 2] = 8
    spot_positions_gene_no = torch.zeros(2).int()
    image_subset = image_numpy[
        spot_positions_yxz[0, 0] - spot_shape[0] // 2 : spot_positions_yxz[0, 0] + spot_shape[0] // 2 + 1,
        [spot_positions_yxz[0, 1]],
        [spot_positions_yxz[0, 2]],
    ]
    image_subset_2 = image_numpy[
        spot_positions_yxz[1, 0] - spot_shape[0] // 2 : spot_positions_yxz[1, 0] + spot_shape[0] // 2 + 1,
        [spot_positions_yxz[1, 1]],
        [spot_positions_yxz[1, 2]],
    ]
    image_subset = torch.sign(torch.asarray(image_subset[np.newaxis]))
    image_subset_2 = torch.sign(torch.asarray(image_subset_2[np.newaxis]))
    expected_mean_spot = torch.concat((image_subset, image_subset_2), dim=0).mean(0)
    mean_spot = compute_mean_spot(image, spot_positions_yxz, spot_positions_gene_no, tile_shape, spot_shape)
    assert torch.allclose(mean_spot, expected_mean_spot)
