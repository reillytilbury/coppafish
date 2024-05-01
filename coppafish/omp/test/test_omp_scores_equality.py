import numpy as np
import pytest


@pytest.mark.pytorch
def test_score_coefficient_image_equality() -> None:
    import torch
    from coppafish.omp import scores, scores_torch

    rng = np.random.RandomState(0)
    spot_shape_shape = (7, 9, 11)
    spot_shape_mean = rng.rand(*spot_shape_shape).astype(np.float32)
    spot_shape_mean[3, 4, 5] = 1
    spot_shape = np.asarray(spot_shape_mean > 0.3, dtype=int)
    spot_shape[spot_shape_mean < 0.1] = -1
    n_genes = 5
    coefs_image_shape = (18, 16, 4, n_genes)
    coefs_image = (100 * (rng.rand(*coefs_image_shape) - 0.5)).astype(np.float32)
    high_coef_bias = 0.2

    scores = scores.score_coefficient_image(coefs_image, spot_shape, spot_shape_mean, high_coef_bias)
    scores_torch = scores_torch.score_coefficient_image(
        torch.asarray(coefs_image),
        torch.asarray(spot_shape),
        torch.asarray(spot_shape_mean),
        high_coef_bias,
        force_cpu=True,
    )
    scores_torch = scores_torch.numpy()

    assert scores.shape == scores_torch.shape
    assert np.allclose(scores, scores_torch, atol=0.075)


if __name__ == "__main__":
    test_score_coefficient_image_equality()
