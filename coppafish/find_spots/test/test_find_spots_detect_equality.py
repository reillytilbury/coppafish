import numpy as np
import pytest


@pytest.mark.pytorch
def test_detect_spots_equality() -> None:
    import torch
    from coppafish.find_spots import detect, detect_torch

    rng = np.random.RandomState(0)

    image_shape = (11, 12, 13)
    image = rng.rand(*image_shape)
    intensity_threshold = 0.01
    radius_xy = 2
    radius_z = 1
    for remove_duplicates in (True, False):
        peak_yxz, peak_intensity = detect.detect_spots(
            image, intensity_threshold, radius_xy, radius_z, remove_duplicates
        )
        peak_yxz_torch, peak_intensity_torch = detect_torch.detect_spots(
            torch.asarray(image), intensity_threshold, radius_xy, radius_z, remove_duplicates
        )
        peak_yxz_torch = peak_yxz_torch.numpy()
        peak_intensity_torch = peak_intensity_torch.numpy()

        assert peak_yxz.shape[1] == image.ndim
        assert np.allclose(peak_yxz, peak_yxz_torch)
        assert np.allclose(peak_intensity, peak_intensity_torch)
