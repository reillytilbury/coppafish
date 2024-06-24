import numpy as np
import torch

from coppafish.find_spots.detect_torch import detect_spots


def test_detect_spots() -> None:
    image_shape = (3, 5, 10)
    image = np.zeros(image_shape).astype(np.float32)
    image[0, 2, 5] = 1
    image[1, 2, 5] = 2
    image[1, 4, 6] = 3

    peak_yxz, peak_intensity = detect_spots(torch.asarray(image).float(), 0.5, 2, 2)
    assert peak_yxz.shape == (2, 3)
    assert torch.allclose(peak_yxz[0], torch.asarray([1, 2, 5])[np.newaxis].int())
    assert torch.allclose(peak_yxz[1], torch.asarray([1, 4, 6])[np.newaxis].int())
    assert peak_intensity.shape == (2,)
    assert torch.allclose(peak_intensity[0], torch.asarray([2]).float())
    assert torch.allclose(peak_intensity[1], torch.asarray([3]).float())
