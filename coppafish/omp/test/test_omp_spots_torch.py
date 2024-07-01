import numpy as np
import torch

from coppafish.omp.spots_torch import compute_mean_spot_from


def test_compute_spots_from() -> None:
    rng = np.random.RandomState(0)
    im_y, im_x, im_z = 10, 11, 12
    image = torch.asarray(rng.rand(im_y, im_x, im_z).astype(np.float32))
    # The most simple case possible.
    spot_positions_yxz = torch.zeros((1, 3)).int()
    spot_positions_yxz[0, 0] = 5
    spot_positions_yxz[0, 1] = 5
    spot_positions_yxz[0, 2] = 6
    spot_shape = (1, 1, 1)
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)
    assert np.allclose(mean_spot, np.sign(image[5, 5, 6]))

    # Two spot positions
    spot_positions_yxz = torch.zeros((2, 3)).int()
    spot_positions_yxz[0, 0] = 5
    spot_positions_yxz[0, 1] = 5
    spot_positions_yxz[0, 2] = 6
    spot_positions_yxz[1, 0] = 4
    spot_positions_yxz[1, 1] = 5
    spot_positions_yxz[1, 2] = 7
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)
    signs = torch.sign(torch.asarray([image[5, 5, 6], image[4, 5, 7]]))
    assert torch.allclose(mean_spot, torch.mean(signs))

    # Two large spot shapes with out of bounds cases
    spot_shape = (15, 11, 3)
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)

    spot_shape = (3, 1, 1)
    spot_positions_yxz = torch.zeros((2, 3)).int()
    spot_positions_yxz[0, 0] = 3
    spot_positions_yxz[0, 1] = 4
    spot_positions_yxz[0, 2] = 1
    spot_positions_yxz[1, 0] = 6
    spot_positions_yxz[1, 1] = 6
    spot_positions_yxz[1, 2] = 8
    image_subset = image[
        spot_positions_yxz[0, 0] - spot_shape[0] // 2 : spot_positions_yxz[0, 0] + spot_shape[0] // 2 + 1,
        [spot_positions_yxz[0, 1]],
        [spot_positions_yxz[0, 2]],
    ]
    image_subset_2 = image[
        spot_positions_yxz[1, 0] - spot_shape[0] // 2 : spot_positions_yxz[1, 0] + spot_shape[0] // 2 + 1,
        [spot_positions_yxz[1, 1]],
        [spot_positions_yxz[1, 2]],
    ]
    image_subset = torch.sign(image_subset[np.newaxis])
    image_subset_2 = torch.sign(image_subset_2[np.newaxis])
    expected_mean_spot = torch.concat((image_subset, image_subset_2), dim=0).mean(0)
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)
    assert torch.allclose(mean_spot, expected_mean_spot)


if __name__ == "__main__":
    test_compute_spots_from()
