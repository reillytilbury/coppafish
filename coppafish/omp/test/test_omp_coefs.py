import torch

from coppafish.omp import coefs


def test_get_next_gene_assignment() -> None:
    n_pixels = 6
    n_rounds_channels_use = 5
    residual_colours = torch.zeros((n_pixels, n_rounds_channels_use), dtype=torch.float32)
    # Pixel 0 should pass score for first gene.
    residual_colours[0, 0] = 1
    # Pixel 1 will contain high scores for two genes, expecting first to be selected.
    residual_colours[1, 0] = 2
    residual_colours[1, 1] = 2
    # Pixel 2 will contain high scores for all genes, expecting it to fail selection.
    residual_colours[2, 0] = 1
    residual_colours[2, 1] = 1
    residual_colours[2, 2] = 1
    residual_colours[2, 3] = 1
    # Pixel 3 contains no intensity, expecting to fail selection.
    # Pixel 4 scores in a gene on the fail list, expecting to fail selection.
    residual_colours[4, 4] = 0.6
    # Pixel 5 scores on fail gene, scores higher on second gene, expecting it to pass.
    residual_colours[5, 1] = 0.7
    residual_colours[5, 4] = 0.6

    all_bled_codes = torch.zeros((4, n_rounds_channels_use), dtype=torch.float32)
    all_bled_codes[0, 0] = 1
    all_bled_codes[1, 1] = 1
    all_bled_codes[2, 2] = 1 / torch.sqrt(torch.tensor(2))
    all_bled_codes[2, 3] = 1 / torch.sqrt(torch.tensor(2))
    all_bled_codes[3, 4] = 1

    fail_gene_indices = torch.ones((n_pixels, 1), dtype=torch.int32)
    fail_gene_indices[:, 0] = 3
    dot_product_threshold = 0.5
    maximum_pass_count = 2

    residual_colours_previous = residual_colours.detach().clone()
    all_bled_codes_previous = all_bled_codes.detach().clone()
    fail_gene_indices_previous = fail_gene_indices.detach().clone()
    best_genes = coefs.get_next_gene_assignment(
        residual_colours=residual_colours,
        all_bled_codes=all_bled_codes,
        fail_gene_indices=fail_gene_indices,
        dot_product_threshold=dot_product_threshold,
        maximum_pass_count=maximum_pass_count,
    )
    assert type(best_genes) is torch.Tensor
    assert best_genes.shape == (n_pixels,), f"Got shape {best_genes.shape}"
    assert best_genes[0] == 0, f"Got {best_genes[0]}"
    assert best_genes[1] == 0
    assert best_genes[2] == coefs.NO_GENE_ASSIGNMENT
    assert best_genes[3] == coefs.NO_GENE_ASSIGNMENT
    assert best_genes[4] == coefs.NO_GENE_ASSIGNMENT
    assert best_genes[5] == 1
    # Since tensors are mutable, check that the inputted tensors have not changed.
    assert torch.allclose(residual_colours_previous, residual_colours)
    assert torch.allclose(all_bled_codes_previous, all_bled_codes)
    assert torch.allclose(fail_gene_indices_previous, fail_gene_indices)


def test_get_next_gene_coefficients() -> None:
    n_pixels = 3
    n_genes_added = 2
    n_rounds_channels_use = 4
    pixel_colours = torch.zeros((n_pixels, n_rounds_channels_use, 1), dtype=torch.float32)
    pixel_colours[0, 0, 0] = 1.3
    pixel_colours[0, 1, 0] = 0.4
    pixel_colours[0, 2, 0] = 0.6
    # The second pixel will have a non-zero residual after genes are fitted.
    pixel_colours[1, 0, 0] = 1 * 0.4
    pixel_colours[1, 3, 0] = 4
    # The third pixel does not quite fit on the second gene.
    pixel_colours[2, 0, 0] = 1 * 0.74
    pixel_colours[2, 1, 0] = 0.4
    pixel_colours[2, 2, 0] = 0.8
    bled_codes = torch.zeros((n_pixels, n_rounds_channels_use, n_genes_added))
    bled_codes[:, 0, 0] = 1
    bled_codes[:, 1, 1] = 0.2
    bled_codes[:, 2, 1] = 0.3
    coefficients, residuals = coefs.get_next_gene_coefficients(pixel_colours=pixel_colours, bled_codes=bled_codes)
    assert type(coefficients) is torch.Tensor
    assert type(residuals) is torch.Tensor
    assert coefficients.shape == (n_pixels, n_genes_added)
    assert residuals.shape == (n_pixels, n_rounds_channels_use)
    abs_tol = 1e-6
    assert torch.isclose(coefficients[0, 0], torch.tensor(1.3).float(), atol=abs_tol)
    assert torch.isclose(coefficients[0, 1], torch.tensor(2).float(), atol=abs_tol)
    assert torch.allclose(residuals[0], torch.tensor(0).float(), atol=abs_tol)
    assert torch.isclose(coefficients[1, 0], torch.tensor(0.4).float(), atol=abs_tol)
    assert torch.isclose(residuals[1, 3], torch.tensor(4).float(), atol=abs_tol)
    assert torch.isclose(residuals[1], torch.tensor(0).float(), atol=abs_tol).sum() == (n_rounds_channels_use - 1)
    assert torch.isclose(coefficients[2, 0], torch.tensor(0.74).float(), atol=abs_tol)
    assert coefficients[2, 1] > 2
    assert residuals[2, 1] < 0
    assert residuals[2, 2] > 0
