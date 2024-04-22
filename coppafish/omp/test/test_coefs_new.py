import numpy as np

from coppafish.omp import coefs_new


def test_compute_omp_coefficients() -> None:
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
        ).toarray()


def test_get_next_best_gene() -> None:
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
    assert best_gene.shape == (im_y, im_x, im_z)
    assert pass_threshold.shape == (im_y, im_x, im_z)
    assert inverse_variance.shape == (im_y, im_x, im_z, n_rounds_channels)
    assert (best_gene[~consider_pixels] == coefs_new.NO_GENE_SELECTION).all()
    assert (~pass_threshold[~consider_pixels]).all()
    assert np.allclose(best_gene[0, 1, 0], 1)
    assert np.allclose(best_gene[0, 2, 0], 2)


def test_weight_selected_genes() -> None:
    im_y, im_x, im_z = 3, 4, 5
    n_rounds_channels = 2
    n_genes = 2
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
    assert coefficients.shape == (im_y, im_x, im_z, n_genes_added)
    assert residuals.shape == (im_y, im_x, im_z, n_rounds_channels)
    assert np.allclose(coefficients[~consider_pixels], 0)
    assert np.allclose(residuals[~consider_pixels], pixel_colours[~consider_pixels])
    assert np.isclose(coefficients[0, 1, 0, 0], 1)
    assert np.isclose(coefficients[0, 1, 0, 1], 0)
    assert np.isclose(coefficients[0, 2, 0, 0], 0)
    assert np.isclose(coefficients[0, 2, 0, 1], 2)

    rng = np.random.RandomState(0)
    weight = rng.rand(im_y, im_x, im_z, n_rounds_channels).astype(np.float32)

    coefficients, residuals = coefs_new.weight_selected_genes(consider_pixels, bled_codes, pixel_colours, genes, weight)
    assert coefficients.shape == (im_y, im_x, im_z, n_genes_added)
    assert residuals.shape == (im_y, im_x, im_z, n_rounds_channels)


if __name__ == "__main__":
    # test_compute_omp_coefficients()
    # test_get_next_best_gene()
    test_weight_selected_genes()
