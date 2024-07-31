import numpy as np

from coppafish.filter.deconvolution import wiener_deconvolve


def test_wiener_deconvolve() -> None:
    rng = np.random.RandomState(0)
    image_shape = (3, 5, 7)
    image = rng.randint(-100, 200, size=image_shape)
    image = image.astype(np.float64)
    im_pad_shape = (2, 3, 4)
    filter = rng.rand(*[image_shape[i] + 2 * im_pad_shape[i] for i in range(3)]).astype(np.complex128)

    deconvolved_image = wiener_deconvolve(image, im_pad_shape, filter)

    assert type(deconvolved_image) is np.ndarray
    assert deconvolved_image.shape == image_shape
