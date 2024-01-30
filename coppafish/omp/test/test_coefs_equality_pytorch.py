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
    assert residual.shape == (n_pixels, n_rounds * n_channels), "Unexpected output residual shape"
    assert coefs.shape == (n_pixels, n_genes_add), "Unexpected output coefs shape"
    residual_torch, coefs_torch = fit_coefs_torch(
        torch.from_numpy(bled_codes), torch.from_numpy(pixel_colors), torch.from_numpy(genes)
    )
    residual_torch = residual_torch.numpy()
    coefs_torch = coefs_torch.numpy()
    assert residual.shape == (n_pixels, n_rounds * n_channels), "Unexpected output residual shape"
    assert coefs.shape == (n_pixels, n_genes_add), "Unexpected output coefs shape"
    assert np.allclose(
        residual, residual_torch, atol=1e-4
    ), "Expected similar residual from optimised and non-optimised OMP"
    assert np.allclose(coefs, coefs_torch, atol=1e-4), "Expected similar coefs from optimised and non-optimised OMP"


@pytest.mark.optimised
def test_fit_coefs_weight_equality_pytorch():
    import torch
    from coppafish.omp.coefs import fit_coefs_weight
    from coppafish.omp.coefs_pytorch import fit_coefs_weight as fit_coefs_weight_torch

    rng = np.random.RandomState(34)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_add = 4
    n_pixels = 9
    bled_codes = rng.rand(n_rounds * n_channels, n_genes) + 1
    pixel_colors = rng.rand(n_rounds * n_channels, n_pixels) + 1
    genes = np.repeat([np.arange(n_genes_add, dtype=int)], n_pixels, axis=0)
    weight = rng.rand(n_pixels, n_rounds * n_channels) + 10
    bled_codes.astype(np.float32)
    pixel_colors.astype(np.float32)
    weight.astype(np.float32)
    residual, coefs = fit_coefs_weight(bled_codes, pixel_colors, genes, weight)
    assert residual.shape == (n_pixels, n_rounds * n_channels), "Unexpected output residual shape"
    assert coefs.shape == (n_pixels, n_genes_add), "Unexpected output coefs shape"
    residual_optimised, coefs_optimised = fit_coefs_weight_torch(
        torch.from_numpy(bled_codes), torch.from_numpy(pixel_colors), torch.from_numpy(genes), torch.from_numpy(weight)
    )
    residual_optimised = residual_optimised.numpy()
    coefs_optimised = coefs_optimised.numpy()
    assert residual_optimised.shape == (n_pixels, n_rounds * n_channels), "Unexpected output residual shape"
    assert coefs_optimised.shape == (n_pixels, n_genes_add), "Unexpected output coefs shape"
    assert np.allclose(
        residual, residual_optimised, atol=1e-4
    ), "Expected similar residual from optimised and non-optimised OMP"
    assert np.allclose(coefs, coefs_optimised, atol=1e-4), "Expected similar coefs from optimised and non-optimised OMP"


test_fit_coefs_weight_equality_pytorch()
