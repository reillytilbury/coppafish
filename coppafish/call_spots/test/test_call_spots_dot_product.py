import numpy as np
import torch

from coppafish.call_spots import dot_product
from coppafish.call_spots import dot_product_pytorch


def test_gene_prob_score():
    # Test that the gene probabilities are different when kappa is varied
    rng = np.random.RandomState(0)
    n_spots = 11
    n_rounds = 3
    n_channels_use = 4
    n_genes = 5
    # Colours range from -1 to 1
    spot_colours = (rng.rand(n_spots, n_rounds, n_channels_use) - 0.5) * 2
    bled_codes = rng.rand(n_genes, n_rounds, n_channels_use)
    kappa_option = 1
    probabilities_1 = dot_product.gene_prob_score(spot_colours, bled_codes)
    probabilities_2 = dot_product.gene_prob_score(spot_colours, bled_codes, kappa_option)
    assert isinstance(probabilities_1, np.ndarray), "Expected ndarray as output"
    assert isinstance(probabilities_2, np.ndarray), "Expected ndarray as output"
    assert probabilities_1.shape == probabilities_2.shape == (n_spots, n_genes), "Expected shape (n_spots, n_genes)"
    assert not np.allclose(probabilities_1, probabilities_2), "Gene probabilities should change as kappa varies"


def test_dot_product_torch():
    # Test the dot product function in pytorch
    rng = np.random.RandomState(0)
    n_spots, n_rounds, n_channels_use, n_genes = 100, 3, 4, 4
    # set gene number for each spot
    gene_no = np.arange(n_spots) % n_genes
    # create gene codes
    bled_codes = np.array([np.roll(np.eye(n_rounds, n_channels_use), i, axis=1) for i in range(n_genes)])
    bled_codes = bled_codes.reshape(n_genes, n_rounds * n_channels_use)
    # add a small amount of noise to the bled codes
    bled_codes += rng.rand(n_genes, n_rounds * n_channels_use) * 0.1
    bled_codes /= np.linalg.norm(bled_codes, axis=1, keepdims=True)
    # create spot colours (with noise)
    spot_colours_noise_free = np.array([bled_codes[gene_no[i]] for i in range(n_spots)])
    noise = rng.rand(n_spots, n_rounds * n_channels_use) * 0.1
    # calculate dot product scores of noise free spot colours and bled codes with no variance terms or norm shift
    gene_prediction, gene_score, gene_score_second, all_score = (
        dot_product_pytorch.dot_product_score(spot_colours=torch.from_numpy(spot_colours_noise_free),
                                              bled_codes=torch.from_numpy(bled_codes))
    )
    assert gene_prediction.shape == gene_score.shape == gene_score_second.shape == (n_spots,), \
        (f"Expected shape {(n_spots, )}. Actual shapes: gene_no: {gene_prediction.shape}, "
         f"gene_score: {gene_score.shape}, gene_score_second: {gene_score_second.shape}")
    assert all_score.shape == (n_spots, n_genes), f"Expected shape {(n_spots, n_genes)}. Actual shape {all_score.shape}"
    assert gene_prediction.numpy().tolist() == gene_no.tolist(), "Expected the same gene numbers"
    assert np.isclose(gene_score.numpy().max(), 1, atol=1e-5), "Expected max gene scores to be 1"

    # calculate dot product scores of noisy spot colours and bled codes with no variance terms or norm shift
    gene_prediction, gene_score, gene_score_second, all_score = (
        dot_product_pytorch.dot_product_score(spot_colours=torch.from_numpy(spot_colours_noise_free + 0.1 * noise),
                                              bled_codes=torch.from_numpy(bled_codes))
    )
    gene_prediction = gene_prediction.numpy()
    assert np.sum(gene_prediction == gene_no)/n_spots > 0.9, "Expected most gene numbers to be correct"
    assert torch.sum(all_score > 1) == 0, "Expected all scores to be less than 1"

    # now add variances
    variance = torch.ones(n_rounds * n_channels_use) * 100 ** 2
    variance[0] = 1
    # repeat the variance for each spot
    variance = variance[None, :].repeat(n_spots, 1)
    gene_prediction, gene_score, gene_score_second, all_score_new = (
        dot_product_pytorch.dot_product_score(spot_colours=torch.from_numpy(spot_colours_noise_free + 0.1 * noise),
                                              bled_codes=torch.from_numpy(bled_codes),
                                              variance=variance)
    )
    gene_prediction = gene_prediction.numpy()
    # we expect this method to only care about r0c0, so the gene score is just going to be
    # score[s, g] = spot_colours[s, 0] / bled_codes[g, 0]
    expected_scores = np.array([spot_colours_noise_free[i, 0] / bled_codes[:, 0] for i in range(n_spots)])
    expected_gene_prediction = np.argmax(expected_scores, axis=1)
    assert np.allclose(gene_prediction, expected_gene_prediction), "Expected gene predictions to be correct"

