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


@pytest.mark.pytorch
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


@pytest.mark.pytorch
def test_get_best_gene_base_equality_pytorch():
    import torch
    from coppafish.omp.coefs import get_best_gene_base
    from coppafish.omp.coefs_pytorch import get_best_gene_base as get_best_gene_base_torch

    rng = np.random.RandomState(98)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    # We test on one pixel because the jax code does a single pixel at a time
    residual_pixel_colors = rng.rand(n_rounds * n_channels)
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    inverse_var = rng.rand(n_rounds * n_channels)
    ignore_genes = np.asarray([1], dtype=int)
    # We add an axis for pixels
    best_gene_1, pass_score_thresh_1 = get_best_gene_base(
        residual_pixel_colors[None], all_bled_codes, norm_shift, score_thresh, inverse_var[None], ignore_genes
    )
    best_gene_2, pass_score_thresh_2 = get_best_gene_base(
        residual_pixel_colors[None], all_bled_codes, norm_shift, score_thresh, inverse_var[None], ignore_genes[None]
    )
    best_gene_optimised_1, pass_score_thresh_optimised_1 = get_best_gene_base_torch(
        torch.from_numpy(residual_pixel_colors[None]),
        torch.from_numpy(all_bled_codes),
        norm_shift,
        score_thresh,
        torch.from_numpy(inverse_var[None]),
        torch.from_numpy(ignore_genes),
    )
    best_gene_optimised_2, pass_score_thresh_optimised_2 = get_best_gene_base_torch(
        torch.from_numpy(residual_pixel_colors[None]),
        torch.from_numpy(all_bled_codes),
        norm_shift,
        score_thresh,
        torch.from_numpy(inverse_var[None]),
        torch.from_numpy(ignore_genes[None]),
    )
    n_pixels = 8
    get_best_gene_base_torch(
        torch.from_numpy(rng.rand(n_pixels, n_rounds * n_channels)),
        torch.from_numpy(all_bled_codes),
        norm_shift,
        score_thresh,
        torch.from_numpy(rng.rand(n_pixels, n_rounds * n_channels)),
        torch.from_numpy(rng.randint(2, size=(n_pixels, 2))),
    )
    best_gene_optimised_1 = best_gene_optimised_1.numpy()
    best_gene_optimised_2 = best_gene_optimised_2.numpy()
    pass_score_thresh_optimised_1 = pass_score_thresh_optimised_1.numpy()
    pass_score_thresh_optimised_2 = pass_score_thresh_optimised_2.numpy()
    assert best_gene_1 == best_gene_2
    assert best_gene_optimised_1 == best_gene_optimised_2
    assert best_gene_1 == best_gene_optimised_1, "Expected the same gene as the result"
    assert pass_score_thresh_1 == pass_score_thresh_2
    assert pass_score_thresh_optimised_1 == pass_score_thresh_optimised_2
    assert pass_score_thresh_1 == pass_score_thresh_optimised_1, "Expected the same boolean pass result"


@pytest.mark.pytorch
def test_get_best_gene_first_iter_equality_pytorch():
    import torch
    from coppafish.omp.coefs import get_best_gene_first_iter
    from coppafish.omp.coefs_pytorch import get_best_gene_first_iter as get_best_gene_first_iter_torch

    rng = np.random.RandomState(60)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_pixels = 9
    residual_pixel_colors = rng.rand(n_pixels, n_rounds * n_channels)
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    background_coefs = rng.rand(n_pixels, n_channels)
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    alpha = rng.rand()
    beta = rng.rand()
    background_genes = rng.randint(n_genes, size=(n_channels))
    best_gene, pass_score_thresh, background_var = get_best_gene_first_iter(
        residual_pixel_colors, all_bled_codes, background_coefs, norm_shift, score_thresh, alpha, beta, background_genes
    )
    assert best_gene.shape == (n_pixels,), "Unexpected shape for `best_gene` output"
    assert pass_score_thresh.shape == (n_pixels,), "Unexpected shape for `pass_score_thresh` output"
    assert background_var.shape == (n_pixels, n_rounds * n_channels), "Unexpected shape for `background_var` output"
    best_gene_optimised, pass_score_thresh_optimised, background_var_optimised = get_best_gene_first_iter_torch(
        torch.from_numpy(residual_pixel_colors),
        torch.from_numpy(all_bled_codes),
        torch.from_numpy(background_coefs),
        norm_shift,
        score_thresh,
        alpha,
        beta,
        torch.from_numpy(background_genes),
    )
    best_gene_optimised = best_gene_optimised.numpy()
    pass_score_thresh_optimised = pass_score_thresh_optimised.numpy()
    background_var_optimised = background_var_optimised.numpy()
    assert best_gene.shape == (n_pixels,), "Unexpected shape for `best_gene` output"
    assert pass_score_thresh.shape == (n_pixels,), "Unexpected shape for `pass_score_thresh` output"
    assert background_var.shape == (n_pixels, n_rounds * n_channels), "Unexpected shape for `background_var` output"
    assert np.allclose(
        best_gene, best_gene_optimised, atol=1e-4
    ), "Expected the same `best_genes` from optimised and non-optimised OMP"
    assert np.allclose(
        pass_score_thresh, pass_score_thresh_optimised, atol=1e-4
    ), "Expected similar `pass_score_thresh` from optimised and non-optimised OMP"
    assert np.allclose(
        background_var, background_var_optimised, atol=1e-4
    ), "Expected similar `background_var` from optimised and non-optimised OMP"


@pytest.mark.pytorch
def test_get_best_gene_equality_pytorch():
    import torch
    from coppafish.omp.coefs import get_best_gene
    from coppafish.omp.coefs_pytorch import get_best_gene as get_best_gene_torch

    rng = np.random.RandomState(131)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_added = 2
    n_pixels = 9
    residual_pixel_colors = rng.rand(n_pixels, n_rounds * n_channels)
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    coefs = rng.rand(n_pixels, n_genes_added)
    genes_added = np.zeros((n_pixels, n_genes_added), dtype=int)
    genes_added[:, 1] = 1
    genes_added[0, 0] = 3
    genes_added[0, 1] = 5
    genes_added[6, 0] = 2
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    alpha = rng.rand()
    background_genes = rng.randint(n_genes, size=(n_channels))
    background_var = rng.rand(n_pixels, n_rounds * n_channels)
    best_gene, pass_score_thresh, inverse_var = get_best_gene(
        residual_pixel_colors,
        all_bled_codes,
        coefs,
        genes_added,
        norm_shift,
        score_thresh,
        alpha,
        background_genes,
        background_var,
    )
    best_gene_optimised, pass_score_thresh_optimised, inverse_var_optimised = get_best_gene_torch(
        torch.from_numpy(residual_pixel_colors),
        torch.from_numpy(all_bled_codes),
        torch.from_numpy(coefs),
        torch.from_numpy(genes_added),
        norm_shift,
        score_thresh,
        alpha,
        torch.from_numpy(background_genes),
        torch.from_numpy(background_var),
    )
    best_gene_optimised = best_gene_optimised.numpy()
    pass_score_thresh_optimised = pass_score_thresh_optimised.numpy()
    inverse_var_optimised = inverse_var_optimised.numpy()
    assert np.allclose(best_gene, best_gene_optimised, atol=1e-4), "Expected the same `best_genes` output"
    assert np.all(pass_score_thresh == pass_score_thresh_optimised), "Expected the same `pass_score_thresh` output"
    assert np.allclose(inverse_var, inverse_var_optimised), "Expected similar `inverse_var` output"


@pytest.mark.pytorch
def test_get_all_coefs_equality():
    import torch
    from coppafish.omp.coefs import get_all_coefs
    from coppafish.omp.coefs_pytorch import get_all_coefs as get_all_coefs_torch

    rng = np.random.RandomState(162)
    n_pixels = 5
    n_rounds = 6
    n_channels = 7
    n_genes = 8
    max_genes = 4
    bled_codes = rng.rand(n_genes, n_rounds, n_channels).astype(np.float32)
    background_shift = rng.rand() * 0.001
    dp_shift = rng.rand() * 0.001
    dp_thresh = rng.rand() * 0.001
    alpha = rng.rand()
    beta = rng.rand()
    for weight_coef_fit in [True, False]:
        pixel_colours = rng.rand(n_pixels, n_rounds, n_channels).astype(np.float32)
        gene_coefs, background_coefs = get_all_coefs(
            pixel_colours, bled_codes, background_shift, dp_shift, dp_thresh, alpha, beta, max_genes, weight_coef_fit
        )
        assert gene_coefs.shape == (n_pixels, n_genes)
        assert background_coefs.shape == (n_pixels, n_channels)
        gene_coefs_optimised, background_coefs_optimised = get_all_coefs_torch(
            torch.from_numpy(pixel_colours),
            torch.from_numpy(bled_codes),
            background_shift,
            dp_shift,
            dp_thresh,
            alpha,
            beta,
            max_genes,
            weight_coef_fit,
        )
        gene_coefs_optimised = gene_coefs_optimised.numpy()
        background_coefs_optimised = background_coefs_optimised.numpy()
        assert gene_coefs_optimised.shape == (n_pixels, n_genes)
        assert background_coefs_optimised.shape == (n_pixels, n_channels)
        assert np.allclose(gene_coefs, gene_coefs_optimised, atol=1e-4), "Expected similar gene coefs"
        assert np.allclose(background_coefs, background_coefs_optimised, atol=1e-4), "Expected similar background coefs"
