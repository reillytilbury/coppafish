import torch


def test_score_coefficient_image() -> None:
    from coppafish.omp import scores_torch

    im_y, im_x, im_z = 4, 5, 6
    spot_shape = 3, 5, 3

    coefficient_image = torch.zeros((im_y, im_x, im_z), dtype=torch.float32)
    coefficient_image[1, 3, 2] = 5
    points = torch.zeros((2, 3), dtype=int)
    points[0, 0] = 1
    points[0, 1] = 3
    points[0, 2] = 2
    spot = torch.zeros(spot_shape, dtype=torch.int16)
    mean_spot = torch.zeros(spot_shape, dtype=torch.float32)
    spot[1, 2, 1] = 1
    mean_spot[1, 2, 1] = 0.5
    spot[1, 3, 2] = 1
    mean_spot[1, 3, 2] = 0.9
    high_bias = 2

    scores = scores_torch.score_coefficient_image(coefficient_image, points, spot, mean_spot, high_bias)

    assert scores.shape == (points.shape[0],)
    assert torch.isclose(scores[0], 5 * 0.5 / ((5 + high_bias) * mean_spot.sum()))
    assert torch.isclose(scores[1], torch.asarray([0], dtype=torch.float32))
