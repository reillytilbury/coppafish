import numpy as np
import torch

from coppafish.find_spots.detect_torch import detect_spots


def test_detect_spots() -> None:
    image_shape = (3, 5, 10)
    image = np.zeros(image_shape).astype(np.float32)
    image[0, 2, 5] = 1
    image[1, 2, 5] = 2
    image[1, 4, 6] = 3

    for remove_duplicates in (True, False):
        peak_yxz, peak_intensity = detect_spots(torch.asarray(image), 0.5, 2, 2, remove_duplicates=remove_duplicates)
        assert peak_yxz.shape == (2, 3)
        assert torch.allclose(peak_yxz[0], torch.asarray([1, 2, 5])[np.newaxis].int())
        assert torch.allclose(peak_yxz[1], torch.asarray([1, 4, 6])[np.newaxis].int())
        assert peak_intensity.shape == (2,)
        assert torch.allclose(peak_intensity[0], torch.asarray([2]).float())
        assert torch.allclose(peak_intensity[1], torch.asarray([3]).float())

    # Check that removing duplicate local maxima is working.
    image_shape = (3, 5, 10)
    image = np.zeros(image_shape).astype(np.int32)
    image[1, 2, 5] = 4
    image[1, 3, 5] = 4
    peak_yxz, peak_intensity = detect_spots(torch.asarray(image), 3.2, 2, 2, remove_duplicates=False)
    assert peak_yxz.shape == (2, 3)
    assert torch.allclose(peak_yxz[0], torch.asarray([1, 2, 5]).int())
    assert torch.allclose(peak_yxz[1], torch.asarray([1, 3, 5]).int())
    assert torch.allclose(peak_intensity, torch.asarray([4]).int())
    peak_yxz, peak_intensity = detect_spots(torch.asarray(image), 3.2, 2, 2, remove_duplicates=True)
    assert peak_yxz.shape == (1, 3)
    assertion = torch.allclose(peak_yxz[0], torch.asarray([1, 2, 5]).int())
    assertion = assertion or torch.allclose(peak_yxz[1], torch.asarray([1, 3, 5]).int())
    assert assertion
    assert torch.allclose(peak_intensity, torch.asarray([4]).int())
