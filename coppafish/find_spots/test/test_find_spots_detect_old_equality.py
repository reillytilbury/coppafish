import numpy as np


def test_detect_spots_equality_old():
    from coppafish.find_spots.detect import detect_spots
    from coppafish.find_spots.detect_old import detect_spots as detect_spots_old

    rng = np.random.RandomState(8)
    n_x = 9
    n_y = 10
    n_z = 11
    image = rng.rand(n_y, n_x, n_z)
    intensity_thresh = rng.rand() * 0.5
    radius_xy = 3
    radius_z = 2
    for remove_duplicates in [True, False]:
        peak_yxz, peak_intensity = detect_spots(image, intensity_thresh, radius_xy, radius_z, remove_duplicates)
        n_peaks = peak_yxz.shape[0]
        assert peak_yxz.shape == (n_peaks, image.ndim)
        assert peak_intensity.shape == (n_peaks,)
        peak_yxz_old, peak_intensity_old = detect_spots_old(
            image, intensity_thresh, radius_xy, radius_z, remove_duplicates
        )
        n_peaks = peak_yxz_old.shape[0]
        assert peak_yxz_old.shape == (n_peaks, image.ndim)
        assert peak_intensity_old.shape == (n_peaks,)
        assert np.allclose(peak_yxz, peak_yxz_old)
        assert np.allclose(peak_intensity, peak_intensity_old)


if __name__ == "__main__":
    test_detect_spots_equality_old()
