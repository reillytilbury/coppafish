import numpy as np

from coppafish.omp.coefs_new import weight_selected_genes


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


if __name__ == "__main__":
    test_weight_selected_genes()
