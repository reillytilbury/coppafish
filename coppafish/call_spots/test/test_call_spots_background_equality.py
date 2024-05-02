import pytest
import numpy as np

from coppafish.call_spots import background, background_pytorch


@pytest.mark.pytorch
def test_fit_background_equality() -> None:
    import torch

    n_spots = 3
    n_rounds = 4
    n_channels = 5
    rng = np.random.RandomState(81)
    # Weighting given to background vector
    bg_weightings = rng.random((n_channels))
    spot_colours = np.ones((n_spots, n_rounds, n_channels))
    for c in range(n_channels):
        spot_colours[:, :, c] *= bg_weightings[c]
    residual, coefficient, bg_vectors = background.fit_background(spot_colours)
    residual_torch, coefficient_torch, bg_vectors_torch = background_pytorch.fit_background(torch.asarray(spot_colours))
    residual_torch = residual_torch.numpy()
    coefficient_torch = coefficient_torch.numpy()
    bg_vectors_torch = bg_vectors_torch.numpy()
    assert np.allclose(residual, residual_torch)
    assert np.allclose(coefficient, coefficient_torch)
    assert np.allclose(bg_vectors, bg_vectors_torch)
