import numpy as np

from coppafish.omp.coefs_new import weight_selected_genes, get_next_best_gene


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

    best_gene, pass_threshold, inverse_variance = get_next_best_gene(
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
    assert (best_gene[~consider_pixels] == -100).all()
    assert (~pass_threshold[~consider_pixels]).all()


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

    coefficients, residuals = weight_selected_genes(consider_pixels, bled_codes, pixel_colours, genes, weight)
    assert coefficients.shape == (im_y, im_x, im_z, n_genes_added)
    assert residuals.shape == (im_y, im_x, im_z, n_rounds_channels)
    assert np.isnan(coefficients[~consider_pixels]).all()
    assert np.isnan(residuals[~consider_pixels]).all()
    assert np.isclose(coefficients[0, 1, 0, 0], 1)
    assert np.isclose(coefficients[0, 1, 0, 1], 0)
    assert np.isclose(coefficients[0, 2, 0, 0], 0)
    assert np.isclose(coefficients[0, 2, 0, 1], 2)

    rng = np.random.RandomState(0)
    weight = rng.rand(im_y, im_x, im_z, n_rounds_channels).astype(np.float32)

    coefficients, residuals = weight_selected_genes(consider_pixels, bled_codes, pixel_colours, genes, weight)
    assert coefficients.shape == (im_y, im_x, im_z, n_genes_added)
    assert residuals.shape == (im_y, im_x, im_z, n_rounds_channels)


if __name__ == "__main__":
    test_get_next_best_gene()
    test_weight_selected_genes()
