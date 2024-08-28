import numpy as np
import torch

from coppafish.find_spots import get_isolated_spots


def test_get_isolated_spots() -> None:
    yxz_positions = np.zeros((5, 3), np.int32)
    yxz_positions[1] = [1, 1, 0]
    yxz_positions[2] = [1, 3, 0]
    yxz_positions[3] = [10, 0, 0]
    yxz_positions[4] = [0, 0, 10]
    distance_threshold_yx = 1.8
    distance_threshold_z = 1.2
    is_isolated = get_isolated_spots(
        yxz_positions=torch.from_numpy(yxz_positions),
        distance_threshold_yx=distance_threshold_yx,
        distance_threshold_z=distance_threshold_z,
    )
    assert type(is_isolated) is torch.Tensor
    assert is_isolated.shape == (5,)
    is_isolated = is_isolated.numpy()
    assert (is_isolated == [False, False, True, True, True]).all(), f"{is_isolated}"
