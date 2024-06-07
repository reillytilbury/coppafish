import numpy as np

from coppafish.omp import scores, scores_torch


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
        scores.score_coefficient_image(coefs_image, spot_shape, spot_shape_mean, high_coef_bias),
        coefs_image / (spot_shape_mean[spot_shape == 1].sum() * (coefs_image + high_coef_bias)),
    )

    n_genes = 21
    coefs_image = np.zeros((3, 3, 1), dtype=np.float32)
    coefs_image[0, 1, 0] = 1
    coefs_image[1, 0, 0] = 1
    coefs_image[1, 1, 0] = 1
    coefs_image[1, 2, 0] = 1
    coefs_image[2, 1, 0] = 1
    # Create a "plus-sign" spot shape with a centre mean of 1
    spot_shape = np.zeros((3, 3, 1), dtype=int)
    spot_shape[0, 1, 0] = 1
    spot_shape[1, 0, 0] = 1
    spot_shape[1, 2, 0] = 1
    spot_shape[2, 1, 0] = 1
    spot_shape[1, 1, 0] = 1
    spot_shape_mean = np.zeros((3, 3, 1), dtype=np.float32)
    spot_shape_mean[spot_shape == 1] = 0.5
    spot_shape_mean[1, 1, 0] = 1
    scores = scores.score_coefficient_image(
        coefs_image[..., np.newaxis].repeat(n_genes, axis=3), spot_shape, spot_shape_mean, 0.25
    )
    expected_scores = np.zeros_like(coefs_image, dtype=np.float32)
    expected_scores[0, 0, 0] = 0.8
    expected_scores[2, 0, 0] = 0.8
    expected_scores[0, 2, 0] = 0.8
    expected_scores[2, 2, 0] = 0.8
    expected_scores[1, 0, 0] = 1.2
    expected_scores[0, 1, 0] = 1.2
    expected_scores[2, 1, 0] = 1.2
    expected_scores[1, 2, 0] = 1.2
    expected_scores[1, 1, 0] = 2.4
    expected_scores /= spot_shape_mean.sum()
    expected_scores = expected_scores[..., np.newaxis].repeat(n_genes, axis=3)
    assert scores.shape == expected_scores.shape
    assert np.allclose(scores, expected_scores)
