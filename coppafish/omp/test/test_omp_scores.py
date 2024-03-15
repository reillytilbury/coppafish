import numpy as np

from coppafish import omp


def test_score_coefficient_image():
    rng = np.random.RandomState(0)
    spot_shape_shape = (7, 9, 11)
    spot_shape_mean = rng.rand(*spot_shape_shape)
    spot_shape_mean[3, 4, 5] = 1
    spot_shape = np.asarray(spot_shape_mean > 0.3, dtype=int)
    spot_shape[spot_shape_mean < 0.1] = -1
    n_genes = 5
    coefs_image_shape = (10, 13, 4, n_genes)
    coefs_image = (100 * (rng.rand(*coefs_image_shape) - 0.5)).astype(np.float32)
    high_coef_bias = 0.2
    scores = omp.scores.score_coefficient_image(coefs_image, spot_shape, spot_shape_mean, high_coef_bias)
    assert scores.shape == coefs_image_shape, f"Scores shape must be {coefs_image_shape}"
    assert (scores >= 0).all(), f"All scores must be >= 0"
    assert (scores <= 1).all(), f"All scores must be >= 1"
    coefs_image = rng.rand(1, 1, 1, n_genes).astype(np.float32)
    assert np.allclose(
        omp.scores.score_coefficient_image(coefs_image, spot_shape, spot_shape_mean, high_coef_bias),
        coefs_image / (spot_shape_mean[spot_shape == 1].sum() * (coefs_image + high_coef_bias)),
    )
