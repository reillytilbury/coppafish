import pytest
import numpy as np


@pytest.mark.pytorch
def test_score_coefficient_image_equality():
    from coppafish.omp.scores import score_coefficient_image
    from coppafish.omp.scores_pytorch import score_coefficient_image as score_coefficient_image_torch

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
    scores = score_coefficient_image(coefs_image, spot_shape, spot_shape_mean, high_coef_bias)
    scores_torch = score_coefficient_image_torch(coefs_image, spot_shape, spot_shape_mean, high_coef_bias)
    assert np.allclose(scores, scores_torch), "Scores do not match"
