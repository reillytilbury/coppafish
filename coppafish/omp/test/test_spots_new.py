import numpy as np

from coppafish.omp.spots_new import compute_mean_spot_from


def test_compute_spots_from() -> None:
    rng = np.random.RandomState(0)
    im_y, im_x, im_z = 10, 11, 12
    image = rng.rand(im_y, im_x, im_z).astype(np.float32)
    # The most simple case possible.
    spot_positions_yxz = np.zeros((1, 3), dtype=np.int16)
    spot_positions_yxz[0] = [5, 5, 6]
    spot_shape = (1, 1, 1)
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)
    assert np.allclose(mean_spot, np.sign(image[5, 5, 6]))

    # Two spot positions
    spot_positions_yxz = np.zeros((2, 3), dtype=np.int16)
    spot_positions_yxz[0] = [5, 5, 6]
    spot_positions_yxz[1] = [4, 5, 7]
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)
    assert np.allclose(mean_spot, np.mean(np.sign([image[5, 5, 6], image[4, 5, 7]])))

    # Two large spot shapes with out of bounds cases
    spot_shape = (15, 11, 3)
    mean_spot = compute_mean_spot_from(image, spot_positions_yxz, spot_shape)
