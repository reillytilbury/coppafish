import numpy as np
import scipy
import torch

from coppafish.omp.spots_torch import compute_mean_spot, is_duplicate_spot, is_true_isolated


def test_is_duplicate_spot() -> None:
    yxz_global_positions = torch.zeros((5, 3)).int()
    yxz_global_positions[0, 0] = 8
    yxz_global_positions[0, 1] = 2
    yxz_global_positions[0, 2] = 2
    yxz_global_positions[2, 0] = -1
    yxz_global_positions[2, 1] = 20
    yxz_global_positions[2, 2] = 2
    yxz_global_positions[3, 0] = -1
    yxz_global_positions[3, 1] = 11
    yxz_global_positions[3, 2] = 0
    yxz_global_positions[4, 0] = 7
    yxz_global_positions[4, 1] = 0
    yxz_global_positions[4, 2] = 1
    tile_number = 0
    tile_centres = torch.zeros((4, 3)).float()
    tile_centres[0, 0] = 7
    tile_centres[0, 1] = 0
    tile_centres[0, 2] = 1
    tile_centres[2, 0] = 0
    tile_centres[2, 1] = 10
    tile_centres[3, 0] = 99_999
    tile_centres[3, 1] = 99_999
    tile_centres[3, 2] = 99_999
    is_duplicate = is_duplicate_spot(yxz_global_positions, tile_number, tile_centres)
    assert type(is_duplicate) is torch.Tensor
    assert is_duplicate.shape == (yxz_global_positions.shape[0],)
    assert not is_duplicate[0]
    assert is_duplicate[1]
    assert is_duplicate[2]
    assert is_duplicate[3]
    assert not is_duplicate[4]
    assert type(yxz_global_positions) is torch.Tensor
    assert type(tile_centres) is torch.Tensor


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
