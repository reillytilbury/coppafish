import pytest
import scipy
import numpy as np
from typing_extensions import assert_type

from coppafish.omp import coefs_new, coefs_torch


@pytest.mark.pytorch
def test_compute_omp_coefficients() -> None:
    import torch

    rng = np.random.RandomState(0)
    im_y, im_x, im_z = 7, 8, 9
    n_rounds_use = 5
    n_channels_use = 6
    n_genes = 2

    pixel_colours = rng.rand(im_y, im_x, im_z, n_rounds_use, n_channels_use).astype(np.float32)
    bled_codes = np.zeros((n_genes, n_rounds_use, n_channels_use), dtype=np.float32)
    bled_codes[0, 0, 0] = 1.2
    bled_codes[1, 0, 1] = 0.5
    bled_codes[1, 0, 2] = 1.5
    maximum_iterations = 3
    background_coefficients = rng.rand(im_y, im_x, im_z, n_channels_use).astype(np.float32) * 0.2
    background_codes = np.zeros((n_channels_use, n_rounds_use, n_channels_use), dtype=np.float32)
    for c in range(n_channels_use):
        background_codes[c, :, c] = 1
    dot_product_threshold = 0.01
    dot_product_norm_shift = rng.rand() * 0.01
    weight_coefficient_fit = [True, False]
    alpha = rng.rand() * 0.02
    beta = rng.rand() * 0.02
    for weight in weight_coefficient_fit:
        pixel_coefficients = coefs_new.compute_omp_coefficients(
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
        assert_type(pixel_coefficients, scipy.sparse.csr_matrix)
        assert pixel_coefficients.shape == (im_y * im_x * im_z, n_genes)
        pixel_coefficients_torch = coefs_torch.compute_omp_coefficients(
            torch.asarray(pixel_colours),
            torch.asarray(bled_codes),
            maximum_iterations,
            torch.asarray(background_coefficients),
            torch.asarray(background_codes),
            dot_product_threshold,
            dot_product_norm_shift,
            weight,
            alpha,
            beta,
        )
        assert_type(pixel_coefficients_torch, scipy.sparse.csr_matrix)
        assert pixel_coefficients_torch.shape == (im_y * im_x * im_z, n_genes)

        assert np.allclose(pixel_coefficients.toarray(), pixel_coefficients_torch.toarray())


@pytest.mark.pytorch
def test_get_next_best_gene_equality() -> None:
    import torch

    im_y, im_x, im_z = 3, 4, 5
    n_rounds_channels = 2
    n_genes = 3
    n_genes_added = 0
    consider_pixels = np.zeros((im_y, im_x, im_z), dtype=bool)
    consider_pixels[0, 1, 0] = True
    consider_pixels[0, 2, 0] = True
    all_bled_codes = np.zeros((n_rounds_channels, n_genes), dtype=np.float32)
    all_bled_codes[:, 0] = 1
    all_bled_codes[0, 1] = 1
    all_bled_codes[1, 2] = 1
    residual_pixel_colours = np.zeros((im_y, im_x, im_z, n_rounds_channels), dtype=np.float32)
    # Give this pixel a weighting of 1 from the first gene
    residual_pixel_colours[0, 1, 0, :] = [1, 0]
    # Give this pixel a weighting of 2 from the second gene
    residual_pixel_colours[0, 2, 0, :] = [0, 2]
    coefficients = np.zeros((im_y, im_x, im_z, n_genes_added), dtype=np.float32)
    genes_added = np.zeros((im_y, im_x, im_z, n_genes_added), dtype=np.int16)
    norm_shift = 0
    score_threshold = 0.1
    alpha = 0.1
    background_genes = np.array([0], dtype=np.int16)
    background_variance = np.ones((im_y, im_x, im_z, n_rounds_channels), dtype=np.float32)

    best_gene, pass_threshold, inverse_variance = coefs_new.get_next_best_gene(
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
    best_gene_torch, pass_threshold_torch, inverse_variance_torch = coefs_torch.get_next_best_gene(
        torch.asarray(consider_pixels),
        torch.asarray(residual_pixel_colours),
        torch.asarray(all_bled_codes),
        torch.asarray(coefficients),
        torch.asarray(genes_added),
        torch.asarray(norm_shift),
        torch.asarray(score_threshold),
        torch.asarray(alpha),
        torch.asarray(background_genes),
        torch.asarray(background_variance),
    )
    best_gene_torch = best_gene_torch.numpy()
    pass_threshold_torch = pass_threshold_torch.numpy()
    inverse_variance_torch = inverse_variance_torch.numpy()

    assert np.allclose(best_gene, best_gene_torch)
    assert np.allclose(pass_threshold, pass_threshold_torch)
    assert np.allclose(inverse_variance, inverse_variance_torch)


@pytest.mark.pytorch
def test_weight_selected_genes_equality() -> None:
    import torch

    im_y, im_x, im_z = 4, 5, 6
    n_rounds_channels = 2
    n_genes = 3
    n_genes_added = 2
    consider_pixels = np.ones((im_y, im_x, im_z), dtype=bool)
    consider_pixels[0, 0, 0] = False
    bled_codes = np.zeros((n_rounds_channels, n_genes), dtype=np.float32)
    bled_codes[0, 0] = 1
    bled_codes[1, 1] = 1
    pixel_colours = np.zeros((im_y, im_x, im_z, n_rounds_channels), dtype=np.float32)
    # Give this pixel a weighting of 1 from the first gene
    pixel_colours[0, 1, 0, :] = [1, 0]
    # Give this pixel a weighting of 2 from the second gene
    pixel_colours[0, 2, 0, :] = [0, 2]
    genes = np.zeros((im_y, im_x, im_z, n_genes_added), dtype=np.int16)
    genes[:, :, :, 1] = 1
    weight = None

    coefficients, residuals = coefs_new.weight_selected_genes(consider_pixels, bled_codes, pixel_colours, genes, weight)
    coefficients_torch, residuals_torch = coefs_torch.weight_selected_genes(
        torch.asarray(consider_pixels),
        torch.asarray(bled_codes),
        torch.asarray(pixel_colours),
        torch.asarray(genes),
        weight,
    )
    coefficients_torch = coefficients_torch.numpy()
    residuals_torch = residuals_torch.numpy()

    assert np.allclose(coefficients, coefficients_torch)
    assert np.allclose(residuals, residuals_torch)

    rng = np.random.RandomState(0)
    weight = rng.rand(im_y, im_x, im_z, n_rounds_channels).astype(np.float32)

    coefficients, residuals = coefs_new.weight_selected_genes(consider_pixels, bled_codes, pixel_colours, genes, weight)
    coefficients_torch, residuals_torch = coefs_torch.weight_selected_genes(
        torch.asarray(consider_pixels),
        torch.asarray(bled_codes),
        torch.asarray(pixel_colours),
        torch.asarray(genes),
        torch.asarray(weight),
    )
    coefficients_torch = coefficients_torch.numpy()
    residuals_torch = residuals_torch.numpy()

    assert np.allclose(coefficients, coefficients_torch)
    assert np.allclose(residuals, residuals_torch)


if __name__ == "__main__":
    test_compute_omp_coefficients()
    test_get_next_best_gene_equality()
    test_weight_selected_genes_equality()
