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

    # calculate dot product scores of noisy spot colours and bled codes with no variance terms or norm shift
    gene_prediction, gene_score, gene_score_second, all_score = (
        dot_product_pytorch.dot_product_score(spot_colours=torch.from_numpy(spot_colours_noise_free + 0.1 * noise),
                                              bled_codes=torch.from_numpy(bled_codes))
    )
    gene_prediction = gene_prediction.numpy()
    assert np.sum(gene_prediction == gene_no)/n_spots > 0.9, "Expected most gene numbers to be correct"

    # calculate dot product scores of noisy spot colours with variance terms weighted highly on gene 0. This should
    # result in gene 0 not being selected for any spot
    variance = np.ones(n_rounds * n_channels_use)
    variance += 1_000 * bled_codes[0]
    variance = np.repeat(variance[None, :], n_spots, axis=0)
    gene_prediction, gene_score, gene_score_second, all_score = (
        dot_product_pytorch.dot_product_score(spot_colours=torch.from_numpy(spot_colours_noise_free + noise),
                                              bled_codes=torch.from_numpy(bled_codes),
                                              variance=torch.from_numpy(variance))
    )
    gene_prediction = gene_prediction.numpy()
    assert np.sum(gene_prediction == 0) == 0, (f"Expected gene 0 to not be selected for any spot, but it was selected "
                                               f"{np.sum(gene_prediction == 0)} times. \n "
                                               f"The scores for these spots are {gene_score[gene_prediction == 0]}. \n"
                                               f"The next best scores are {gene_score_second[gene_prediction == 0]}, \n"
                                               f"and the weighted colours are \n "
                                               f"{((spot_colours_noise_free + noise)[gene_prediction == 0] / np.sqrt(variance[gene_prediction == 0])).reshape(-1, n_rounds, n_channels_use)}")
