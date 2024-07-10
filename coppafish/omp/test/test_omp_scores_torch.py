import torch


def test_score_coefficient_image() -> None:
    from coppafish.omp import scores_torch

    im_y, im_x, im_z = 4, 5, 6
    spot_shape = 1, 3, 5

    coefficient_image = torch.zeros((1, im_y, im_x, im_z), dtype=torch.float32)
    coefficient_image[0, 1, 3, 2] = 5
    points = torch.zeros((2, 3), dtype=int)
    points[0, 0] = 1
    points[0, 1] = 3
    points[0, 2] = 2
    spot = torch.zeros(spot_shape, dtype=torch.int16)
    mean_spot = torch.zeros(spot_shape, dtype=torch.float32)
    spot[0, 1, 2] = 1
    mean_spot[0, 1, 2] = 0.5
    spot[0, 2, 2] = 1
    mean_spot[0, 2, 2] = 0.9
    mean_spot[0, 1, 3] = 0.1

    scores = scores_torch.score_coefficient_image(coefficient_image, spot, mean_spot)

    assert scores.shape == coefficient_image.shape
    assert torch.isclose(scores[0, 1, 3, 2], 5 * 0.5 / mean_spot[spot == 1].sum())
    assert torch.isclose(scores[0, 0, 0, 0], torch.asarray([0], dtype=torch.float32))
