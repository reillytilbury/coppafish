import numpy as np
import torch

from coppafish.omp import coefs_torch


def test_compute_omp_coefficients() -> None:
    torch.manual_seed(0)
    im_y, im_x, im_z = 7, 8, 9
    n_pixels = im_y * im_x * im_z
    n_rounds_use = 5
    n_channels_use = 6
    n_genes = 2

    pixel_colours = torch.rand((n_pixels, n_rounds_use * n_channels_use)).float()
    bled_codes = torch.zeros((n_genes, n_rounds_use, n_channels_use), dtype=torch.float32)
    bled_codes[0, 0, 0] = 1.2
    bled_codes[1, 0, 1] = 0.5
    bled_codes[1, 0, 2] = 1.5
    bled_codes = bled_codes.reshape((n_genes, n_rounds_use * n_channels_use))
    pixel_colours[0] += bled_codes[0]
    pixel_colours[0] += 2 * bled_codes[1]
    maximum_iterations = 3
    background_coefficients = torch.rand((n_pixels, n_channels_use)).float() * 0.002
    background_codes = torch.zeros((n_channels_use, n_rounds_use, n_channels_use), dtype=torch.float32)
    for c in range(n_channels_use):
        background_codes[c, :, c] = 1
    background_codes = background_codes.reshape((n_channels_use, n_rounds_use * n_channels_use))
    dot_product_threshold = 0.0001
    dot_product_norm_shift = torch.rand(1).item() * 0.01
    weight_coefficient_fit = [True, False]
    alpha = torch.rand(1).item() * 0.02
    beta = torch.rand(1).item() * 0.02
    for weight in weight_coefficient_fit:
        pixel_coefficients = coefs_torch.compute_omp_coefficients(
            pixel_colours,
            bled_codes,
            maximum_iterations,
            background_coefficients,
            background_codes,
            dot_product_threshold,
            dot_product_norm_shift,
            weight,
            alpha,
            beta,
        )
        assert type(pixel_coefficients) is torch.Tensor
        assert pixel_coefficients.shape == (n_pixels, n_genes)
        # Run with a super high iteration to check early stopping works.
        pixel_coefficients = coefs_torch.compute_omp_coefficients(
            pixel_colours,
            bled_codes,
            1_000,
            background_coefficients,
            background_codes,
            dot_product_threshold,
            dot_product_norm_shift,
            weight,
            alpha,
            beta,
        )
        assert type(pixel_coefficients) is torch.Tensor
        assert pixel_coefficients.shape == (n_pixels, n_genes)


def test_get_next_best_gene() -> None:
    torch.manual_seed(0)
    im_y, im_x, im_z = 3, 4, 5
    n_pixels = im_y * im_x * im_z
    n_rounds_channels = 2
    n_genes = 3
    n_genes_added = 0
    consider_pixels = torch.zeros(n_pixels, dtype=bool)
    consider_pixels[1] = True
    consider_pixels[2] = True
    all_bled_codes = torch.zeros((n_rounds_channels, n_genes), dtype=torch.float32)
    all_bled_codes[:, 0] = 1
    all_bled_codes[0, 1] = 1
    all_bled_codes[1, 2] = 1
    residual_pixel_colours = torch.zeros((n_pixels, n_rounds_channels), dtype=torch.float32)
    # Give this pixel a weighting of 1 from the first gene
    residual_pixel_colours[1, 0] = 1
    residual_pixel_colours[1, 1] = 0
    # Give this pixel a weighting of 2 from the second gene
    residual_pixel_colours[2, 0] = 0
    residual_pixel_colours[2, 1] = 2
    residual_pixel_colours = residual_pixel_colours.reshape((n_pixels, n_rounds_channels))
    coefficients = torch.zeros((n_pixels, n_genes_added), dtype=torch.float32)
    genes_added = torch.zeros((n_pixels, n_genes_added), dtype=torch.int16)
    norm_shift = 0
    score_threshold = 0.1
    alpha = 0.1
    background_genes = torch.asarray([0], dtype=torch.int16)
    background_variance = torch.ones((n_pixels, n_rounds_channels), dtype=torch.float32)

    best_gene, pass_threshold, inverse_variance = coefs_torch.get_next_best_gene(
        consider_pixels,
        residual_pixel_colours,
        all_bled_codes,
        coefficients,
        genes_added,
        norm_shift,
        score_threshold,
        alpha,
        background_genes,
        background_variance,
    )
    assert best_gene.shape == (n_pixels,)
    assert pass_threshold.shape == (n_pixels,)
    assert inverse_variance.shape == (n_pixels, n_rounds_channels)
    assert (best_gene[~consider_pixels] == coefs_torch.NO_GENE_SELECTION).all()
    assert (~pass_threshold[~consider_pixels]).all()
    assert np.allclose(best_gene[1], 1)
    assert np.allclose(best_gene[2], 2)


def test_weight_selected_genes() -> None:
    torch.manual_seed(0)
    im_y, im_x, im_z = 4, 5, 6
    n_pixels = im_y * im_x * im_z
    n_rounds_channels = 2
    n_genes = 3
    n_genes_added = 2
    consider_pixels = torch.ones(n_pixels, dtype=bool)
    consider_pixels[0] = False
    bled_codes = torch.zeros((n_rounds_channels, n_genes), dtype=torch.float32)
    bled_codes[0, 0] = 1
    bled_codes[1, 1] = 1
    pixel_colours = torch.zeros((n_pixels, n_rounds_channels), dtype=torch.float32)
    # Give this pixel a weighting of 1 from the first gene
    pixel_colours[1, 0] = 1
    pixel_colours[1, 1] = 0
    # Give this pixel a weighting of 2 from the second gene
    pixel_colours[2, 0] = 0
    pixel_colours[2, 1] = 2
    genes = torch.zeros((n_pixels, n_genes_added), dtype=torch.int16)
    genes[:, 1] = 1
    weight = None

    coefficients, residuals = coefs_torch.weight_selected_genes(
        consider_pixels, bled_codes, pixel_colours, genes, weight
    )
    assert coefficients.shape == (n_pixels, n_genes_added)
    assert residuals.shape == (n_pixels, n_rounds_channels)
    assert np.allclose(coefficients[~consider_pixels], 0)
    assert np.allclose(residuals[~consider_pixels], pixel_colours[~consider_pixels])
    assert np.isclose(coefficients[1, 0].numpy(), 1)
    assert np.isclose(coefficients[1, 1].numpy(), 0)
    assert np.isclose(coefficients[2, 0].numpy(), 0)
    assert np.isclose(coefficients[2, 1].numpy(), 2)

    weight = torch.rand((n_pixels, n_rounds_channels)).float()

    coefficients, residuals = coefs_torch.weight_selected_genes(
        consider_pixels, bled_codes, pixel_colours, genes, weight
    )
    assert coefficients.shape == (n_pixels, n_genes_added)
    assert residuals.shape == (n_pixels, n_rounds_channels)
