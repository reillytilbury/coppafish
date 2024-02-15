import pytest
import numpy as np


@pytest.mark.optimised
def test_get_shifts_from_kernel_equality():
    from coppafish.utils.morphology.filter import get_shifts_from_kernel
    from coppafish.utils.morphology.filter_optimised import get_shifts_from_kernel as get_shifts_from_kernel_jax

    rng = np.random.RandomState(10)
    kernel = rng.randint(-10, 11, size=(11, 10, 7))
    output = get_shifts_from_kernel(kernel)
    output_jax = get_shifts_from_kernel_jax(kernel)
    for i in range(len(output)):
        assert np.all(output[i] == output_jax[i])


@pytest.mark.optimised
def test_imfilter_coords_equality():
    from coppafish.utils.morphology.filter import imfilter_coords
    from coppafish.utils.morphology.filter_optimised import imfilter_coords as imfilter_coords_jax

    rng = np.random.RandomState(0)
    n_points = 10

    image_shape_2d = (50, 50)
    image_2d = rng.randint(100, size=image_shape_2d)
    kernel_2d = rng.randint(2, size=(5, 5))
    coords_2d = rng.randint(50, size=(n_points, 2))
    result = imfilter_coords(image_2d, kernel_2d, coords_2d)
    result_jax = imfilter_coords_jax(image_2d, kernel_2d, coords_2d)
    assert result.shape == result_jax.shape == (n_points,), "Expected output shape to be (n_points, )"
    assert (result == result_jax).all(), "Outputs are not the same"

    image_shape_3d = (25, 25, 4)
    image_3d = rng.rand(100, size=image_shape_3d)
    kernel_3d = rng.randint(2, size=(5, 5, 2))
    coords_3d = rng.randint(50, size=(n_points, 3))
    result = imfilter_coords(image_3d, kernel_3d, coords_3d)
    result_jax = imfilter_coords_jax(image_3d, kernel_3d, coords_3d)
    assert result.shape == result_jax.shape == (n_points,), "Expected output shape to be (n_points, )"
    assert (result == result_jax).all(), "Outputs are not the same"
