import pytest
import numpy as np


@pytest.mark.pytorch
def test_dot_product_score_equality():
    import torch
    from coppafish.call_spots.dot_product import dot_product_score
    from coppafish.call_spots.dot_product_pytorch import dot_product_score as dot_product_score_torch

    rng = np.random.RandomState(7)
    n_spots = 2
    n_rounds = 7
    n_genes = 4
    n_channels_use = 3
    spot_colours = rng.rand(n_spots, n_rounds * n_channels_use)
    bled_codes = rng.rand(n_genes, n_rounds * n_channels_use)
    bled_codes /= np.linalg.norm(bled_codes, axis=1, keepdims=True)
    for weight_squared in [rng.rand(n_spots, n_rounds * n_channels_use), None]:
        for norm_shift in [rng.rand()]:
            gene_no, gene_score, gene_score_second, score = dot_product_score(
                spot_colours, bled_codes, weight_squared, norm_shift
            )
            assert gene_no.shape == (n_spots,), f"Expected `gene_no` to have shape ({n_spots}, )"
            assert gene_score.shape == (n_spots,), f"Expected `gene_score` to have shape ({n_spots}, )"
            gene_no_torch, gene_score_torch, gene_score_second_torch, score_torch = dot_product_score_torch(
                torch.from_numpy(spot_colours),
                torch.from_numpy(bled_codes),
                torch.from_numpy(weight_squared) if weight_squared is not None else None,
                norm_shift,
            )
            gene_no_torch = gene_no_torch.numpy()
            gene_score_torch = gene_score_torch.numpy()
            gene_score_second_torch = gene_score_second_torch.numpy()
            score_torch = score_torch.numpy()
            assert score_torch.shape == (n_spots, n_genes), "Unexpected dot product score shape"
            assert np.allclose(gene_score, np.max(score_torch, axis=1), atol=1e-4), "Expected best scores to be similar"
            assert np.allclose(gene_no, np.argmax(score_torch, axis=1)), "Expected the same best gene numbers"
            assert np.allclose(score, score_torch, atol=1e-4)
            assert np.allclose(gene_no, gene_no_torch, atol=1e-4)
            assert np.allclose(gene_score, gene_score_torch, atol=1e-4)
            assert np.allclose(gene_score_second, gene_score_second_torch, atol=1e-4)
            assert np.allclose(score, score_torch, atol=1e-4)
