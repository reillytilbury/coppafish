import numpy as np

from coppafish import omp


def test_score_omp_spots():
    rng = np.random.RandomState(0)
    spot_shape_shape = (7, 9, 11)
    spot_shape_mean = rng.rand(*spot_shape_shape)
    spot_shape_mean[3, 4, 5] = 1
    spot_shape = np.asarray(spot_shape_mean > 0.1, dtype=int)
    n_pixels = 1
    n_genes = 5
    pixel_yxz = np.zeros((n_pixels, 3), dtype=int)
    pixel_coefs = rng.rand(n_pixels, n_genes)
    sigmoid_weight = 0.25
    scores = omp.scores.score_omp_spots(spot_shape, spot_shape_mean, pixel_yxz, pixel_coefs, sigmoid_weight)
    assert np.allclose(
        scores,
        omp.scores.score_omp_spots(spot_shape, spot_shape_mean, pixel_yxz, pixel_coefs, sigmoid_weight, np.array([0])),
    )
    assert scores.shape == (n_pixels, n_genes), "Expected OMP scores to be shape (n_pixels, n_genes)"
    assert (scores >= 0).all() and (scores <= 1).all(), "Expected all OMP scores to be between 0 and 1 inclusive"
    zero_coef_score = 1 / (1 + sigmoid_weight)
    expected_scores = (spot_shape_mean[spot_shape == 1].sum() - 1) * zero_coef_score + 1 / (
        1 + sigmoid_weight * np.exp(-pixel_coefs[0])
    )
    expected_scores /= spot_shape_mean[spot_shape == 1].sum()
    assert np.allclose(scores[0], expected_scores)
