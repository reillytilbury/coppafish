import pytest
import numpy as np


@pytest.mark.pytorch
def test_wiener_deconvolve_equality():
    rng = np.random.RandomState(0)
    im_y = 20
    im_x = 20
    im_z = 10
    image = rng.randint(-200, 200, size=(im_y, im_x, im_z), dtype=int)
    im_pad_shape = [1, 1, 2]
    filter = rng.rand(im_y + 2, im_x + 2, im_z + 4)
    from coppafish.filter.deconvolution_pytorch import wiener_deconvolve

    output_2 = wiener_deconvolve(image, im_pad_shape, filter)
    from coppafish.filter.deconvolution import wiener_deconvolve

    output = wiener_deconvolve(image, im_pad_shape, filter)
    assert np.allclose(output, output_2)


if __name__ == "__main__":
    test_wiener_deconvolve_equality()
