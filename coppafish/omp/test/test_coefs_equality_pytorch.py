import pytest
import numpy as np


@pytest.mark.pytorch
def test_fit_coefs_equality_pytorch():
    import torch
    from coppafish.omp.coefs import fit_coefs
    from coppafish.omp.coefs_pytorch import fit_coefs as fit_coefs_torch
    
    # We want 1 test that the function `fit_coefs` is giving similar results in the pytorch and non-pytorch code
    rng = np.random.RandomState(9)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_add = 13
    n_pixels = 9
    bled_codes = rng.rand(n_rounds * n_channels, n_genes)
    pixel_colors = rng.rand(n_rounds * n_channels, n_pixels)
    genes = rng.randint(n_genes, size=(n_pixels, n_genes_add))
    residual, coefs = fit_coefs(bled_codes, pixel_colors, genes)
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    residual_torch, coefs_torch = fit_coefs_torch(
        torch.from_numpy(bled_codes), torch.from_numpy(pixel_colors), torch.from_numpy(genes)
    )
    residual_torch = residual_torch.numpy()
    coefs_torch = coefs_torch.numpy()
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    assert np.allclose(residual, residual_torch, atol=1e-4), \
        'Expected similar residual from optimised and non-optimised OMP'
    assert np.allclose(coefs,    coefs_torch,    atol=1e-4), \
        'Expected similar coefs from optimised and non-optimised OMP'
