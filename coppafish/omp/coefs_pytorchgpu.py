import os
import tqdm
import torch
import scipy
import numpy as np
from typing import Tuple, Union, List, Any

from . import base
from .. import utils, call_spots, log
from ..call_spots import dot_product_pytorch as dot_product
from ..setup import NotebookPage


def fit_coefs(
    bled_codes: torch.Tensor, pixel_colors: torch.Tensor, genes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`. Can
    also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes (`((n_rounds * n_channels) x n_genes) tensor[float]`): flattened then transposed bled codes which
            usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors (`((n_rounds * n_channels) x n_pixels) tensor[float]` if `n_genes==1` otherwise
            `(n_rounds * n_channels) tensor[float]`): flattened then transposed pixel colors which usually has the
            shape `[n_pixels x n_rounds x n_channels]`.
        genes: `(n_pixels x n_genes_add) tensor[int]`: indices of codes in bled_codes to find coefficients for which
            best explain each pixel_color.

    Returns:
        - `(n_pixels x (n_rounds * n_channels)) tensor[float]`: residual pixel_colors after removing bled_codes with
            coefficients specified by coef.
        - (`(n_pixels x n_genes_add) tensor[float]` if n_genes == 1, `n_genes tensor[float]` if n_pixels == 1):
            coefficient found through least squares fitting for each gene.
    """
    n_pixels = pixel_colors.shape[1]
    residuals = torch.zeros((n_pixels, pixel_colors.shape[0]))

    # The arguments given are of shapes (n_pixels, (n_rounds * n_channels), n_genes_add) and
    # (n_pixels, (n_rounds * n_channels), 1). Pytorch then knows to batch over pixels
    # Coefs is shape (n_pixels, n_genes_add)
    # We use the driver "gels" here because this is the only option for GPU torch. This assumes that
    # bled_codes_weighted is full-rank, which I think it always is in real data examples (at least it very well should
    # be!)
    coefs = torch.linalg.lstsq(
        bled_codes[:, genes].transpose(0, 1),
        pixel_colors.T[..., None],
        rcond=None,
        driver="gels",
    )[0][:, :, 0]
    # Using pytorch's handy batch matrix multiply to batch over n_pixels for optimised code
    residuals = torch.swapaxes(
        pixel_colors
        - torch.swapaxes(torch.bmm(torch.swapaxes(bled_codes[:, genes], 0, 1), coefs[..., None])[..., 0], 0, 1),
        0,
        1,
    )

    return residuals.type(torch.float32), coefs.type(torch.float32)


def fit_coefs_weight(
    bled_codes: torch.Tensor, pixel_colors: torch.Tensor, genes: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s. The `weight` indicates which rounds/channels should
    have more influence when finding the coefficients of each gene.

    Args:
        bled_codes (`((n_rounds * n_channels) x n_genes) tensor[float]`): flattened then transposed bled codes which
            usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors (`((n_rounds * n_channels) x n_pixels) tensor[float]`): flattened then transposed `pixel_colors`
            which usually has the shape `(n_pixels x n_rounds x n_channels)`.
        genes: `(n_pixels x n_genes_add) tensor[int]`: indices of codes in bled_codes to find coefficients for which
            best explain each pixel_color.
        weight: (`(n_pixels x (n_rounds * n_channels)) tensor[float]`: `weight[s, i]` is the weight to be applied to
            round_channel `i` when computing coefficient of each `bled_code` for pixel `s`.

    Returns:
        - residual (`(n_pixels x (n_rounds * n_channels)] tensor[float32]`): residual pixel_colors after removing
            bled_codes with coefficients specified by coefs.
        - coefs - (`[n_pixels x n_genes_add] tensor[float32]`): coefficients found through least squares fitting for
            each gene.
    """
    # (n_pixels, n_rounds_channels, n_genes_add)
    bled_codes_weighted = bled_codes[:, genes].swapaxes(0, 1) * weight[..., None]
    del bled_codes, genes
    # (n_pixels, n_rounds_channels)
    pixel_colors_weighted = pixel_colors.T * weight
    del pixel_colors
    # We use the driver "gels" here because this is the only option for GPU torch. This assumes that
    # bled_codes_weighted is full-rank, which I think it always is in real data examples (at least it very well should
    # be!)
    coefs = torch.linalg.lstsq(bled_codes_weighted, pixel_colors_weighted, rcond=-1, driver="gels")[0]
    # Using pytorch's batch matrix multiplication to eliminate a need for a for loop
    residuals = pixel_colors_weighted - torch.bmm(bled_codes_weighted, coefs[..., None])[..., 0]
    residuals = residuals / weight

    return residuals.type(torch.float32), coefs.type(torch.float32)


def get_best_gene_base(
    residual_pixel_colours: torch.Tensor,
    all_bled_codes: torch.Tensor,
    norm_shift: float,
    score_thresh: float,
    inverse_var: torch.Tensor,
    ignore_genes: torch.Tensor,
) -> Tuple[int, bool]:
    """
    Computes the `dot_product_score` between `residual_pixel_color` and each code in `all_bled_codes`. If `best_score`
    is less than `score_thresh` or if the corresponding `best_gene` is in `ignore_genes`, then `pass_score_thresh` will
    be False.

    Args:
        residual_pixel_colours (`(n_pixels x (n_rounds * n_channels)) tensor[float]`): residual pixel colors from
            previous iteration of omp.
        all_bled_codes (`[n_genes x (n_rounds * n_channels)] tensor[float]`): `bled_codes` such that `spot_color` of a
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes
            and background.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be
            added in the current iteration.
        inverse_var (`(n_pixels x (n_rounds * n_channels)) tensor[float]`): inverse of variance in each round/channel
            for each pixel based on genes fit on previous iteration. Used as `weight_squared` when computing
            `dot_product_score`.
        ignore_genes (`(n_genes_ignore) or (n_pixels x n_genes_ignore) tensor[int]`): if `best_gene` is one of these,
            `pass_score_thresh` will be `False`. If no pixel axis, then the same genes are ignored for each pixel
            (useful for the first iteration of OMP edge case).

    Returns:
        - best_genes (n_pixels tensor[int]): The best gene to add next for each pixel.
        - pass_score_threshes (n_pixels tensor[bool]): `True` if `best_score > score_thresh` and `best_gene` not in
            `ignore_genes`.
    """
    assert residual_pixel_colours.ndim == 2, "`residual_pixel_colors` must be two dimensional"
    assert all_bled_codes.ndim == 2, "`all_bled_codes` must be two dimensional"
    assert inverse_var.ndim == 2, "`inverse_var` must be two dimensional"
    assert ignore_genes.ndim == 1 or ignore_genes.ndim == 2, "`ignore_genes` must be one or two dimensional"
    n_pixels = residual_pixel_colours.shape[0]
    n_genes_ignore = ignore_genes.shape[-1]
    if ignore_genes.ndim == 2:
        assert ignore_genes.shape[0] == n_pixels, "`ignore_genes` must have n_pixels in first axis if two dimensional"

    # Calculate score including background genes as if best gene is background, then stop iteration. all_scores has
    # shape (n_pixels, n_genes)
    all_scores = dot_product.dot_product_score(residual_pixel_colours, all_bled_codes, inverse_var, norm_shift)[3]
    del residual_pixel_colours, all_bled_codes, inverse_var
    # best_genes has shape (n_pixels, )
    best_genes = torch.argmax(torch.abs(all_scores), dim=1)
    # Take the best gene score for each pixel.
    best_scores = all_scores[range(n_pixels), best_genes]
    del all_scores
    # If best_gene is in ignore_genes, set score below score_thresh, i.e. set the score to zero.
    if ignore_genes.ndim == 1:
        best_scores *= torch.isin(best_genes, ignore_genes, invert=True)
    else:
        best_scores *= torch.logical_not(
            (torch.repeat_interleave(best_genes[:, None], n_genes_ignore, dim=1) == ignore_genes).any(1)
        )
    pass_score_threshes = torch.abs(best_scores) > score_thresh
    return best_genes, pass_score_threshes


def get_best_gene_first_iter(
    residual_pixel_colors: torch.Tensor,
    all_bled_codes: torch.Tensor,
    background_coefs: torch.Tensor,
    norm_shift: float,
    score_thresh: float,
    alpha: float,
    beta: float,
    background_genes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds the `best_gene` to add next based on the dot product score with each `bled_code`.
    If `best_gene` is in `background_genes` or `best_score < score_thresh` then `pass_score_thresh = False`.
    Different for first iteration as no actual non-zero gene coefficients to consider when computing variance
    or genes that can be added which will cause `pass_score_thresh` to be `False`.

    Args:
        residual_pixel_colors (`(n_pixels x (n_rounds * n_channels)) tensor[float]`): residual pixel color from
            previous iteration of omp.
        all_bled_codes (`(n_genes x (n_rounds * n_channels)) tensor[float]`): `bled_codes` such that `spot_color` of a
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes
            and background.
        background_coefs (`(n_pixels x n_channels) tensor[float]`): `coefs[g]` is the weighting for gene
            `background_genes[g]` found by the omp algorithm. All are non-zero.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be
            added in the current iteration.
        alpha (float): Used for `fitting_variance`, by how much to increase variance as genes added.
        beta (float): Used for `fitting_variance`, the variance with no genes added (`coef=0`) is `beta**2`.
        background_genes (`(n_channels) tensor[int]`): Indices of codes in `all_bled_codes` which correspond to
            background. If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]`
            will be `False`.

    Returns:
        - best_gene (`(n_pixels) tensor[int]`): `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh (`(n_pixels) tensor[bool]`): true if `best_score > score_thresh`.
        - background_var (`(n_pixels x (n_rounds * n_channels)) tensor[float]`): variance in each round/channel based
            on just the background.

    Notes:
        - The returned tensors are placed on the same device as the parameters' device.
    """
    device = residual_pixel_colors.device
    n_pixels = residual_pixel_colors.shape[0]
    best_genes = torch.zeros(n_pixels, dtype=int, device=device)
    pass_score_threshes = torch.zeros(n_pixels, dtype=bool, device=device)
    # Ensure bled_codes are normalised for each gene
    all_bled_codes /= all_bled_codes.norm(dim=1, keepdim=True)
    background_vars = torch.square(background_coefs) @ torch.square(all_bled_codes[background_genes]) * alpha + beta**2
    best_genes, pass_score_threshes = get_best_gene_base(
        residual_pixel_colors, all_bled_codes, norm_shift, score_thresh, 1 / background_vars, background_genes
    )

    return best_genes, pass_score_threshes, background_vars.type(torch.float32)


def get_best_gene(
    residual_pixel_colors: torch.Tensor,
    all_bled_codes: torch.Tensor,
    coefs: torch.Tensor,
    genes_added: torch.Tensor,
    norm_shift: float,
    score_thresh: float,
    alpha: float,
    background_genes: torch.Tensor,
    background_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.
    If `best_gene[s]` is in `background_genes`, already in `genes_added[s]` or `best_score[s] < score_thresh`,
    then `pass_score_thresh[s] = False`.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds * n_channels)]`.
            Residual pixel colors from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds * n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        coefs: `float [n_pixels x n_genes_added]`.
            `coefs[s, g]` is the weighting of pixel `s` for gene `genes_added[g]` found by the omp algorithm on previous
            iteration. All are non-zero.
        genes_added: `int [n_pixels x n_genes_added]`
            Indices of genes added to each pixel from previous iteration of omp.
            If the best gene for pixel `s` is set to one of `genes_added[s]`, `pass_score_thresh[s]` will be False.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_variance`, by how much to increase variance as genes added.
        background_genes: `int [n_channels]`.
            Indices of codes in all_bled_codes which correspond to background.
            If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]` will be False.
        background_var: `float [n_pixels x (n_rounds * n_channels)]`.
            Contribution of background genes to variance (which does not change throughout omp iterations)  i.e.
            `background_coefs**2 @ all_bled_codes[background_genes]**2 * alpha + beta ** 2`.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh - `bool [n_pixels]`.
            `True` if `best_score > score_thresh`.
        - inverse_var - `float [n_pixels x (n_rounds * n_channels)]`.
            Inverse of variance of each pixel in each round/channel based on genes fit on previous iteration.
            Includes both background and gene contribution.

    Notes:
        - The variance computed is based on maximum likelihood estimation - it accounts for all genes and background
            fit in each round/channel. The more genes added, the greater the variance so if the inverse is used as a
            weighting for omp fitting or choosing the next gene, the rounds/channels which already have genes in will
            contribute less.
        - Returned variables are on the same device as the parameters' device.
    """
    assert residual_pixel_colors.ndim == 2
    assert all_bled_codes.ndim == 2
    assert coefs.ndim == 2
    assert genes_added.ndim == 2
    assert background_genes.ndim == 1

    n_pixels, n_rounds_channels = residual_pixel_colors.shape
    best_genes = torch.zeros((n_pixels), dtype=int)
    pass_score_threshes = torch.zeros((n_pixels), dtype=bool)
    inverse_vars = torch.zeros((n_pixels, n_rounds_channels), dtype=torch.float32)
    # Ensure bled_codes are normalised for each gene
    all_bled_codes /= all_bled_codes.norm(dim=1, keepdim=True)

    # See Josh's OMP documentation for details about this exact equation
    inverse_vars = torch.reciprocal(
        torch.bmm((coefs**2)[:, None], all_bled_codes[genes_added] ** 2 * alpha)[:, 0] + background_var
    )
    del coefs, background_var
    # Similar function to numpy's .repeat
    ignore_genes = torch.repeat_interleave(background_genes[None], n_pixels, dim=0)
    ignore_genes = torch.concatenate((ignore_genes, genes_added), dim=1)
    # calculate score including background genes as if best gene is background, then stop iteration.
    best_genes, pass_score_threshes = get_best_gene_base(
        residual_pixel_colors,
        all_bled_codes,
        norm_shift,
        score_thresh,
        inverse_vars,
        ignore_genes,
    )

    return best_genes, pass_score_threshes, inverse_vars


def get_all_coefs(
    pixel_colors: torch.Tensor,
    bled_codes: torch.Tensor,
    background_shift: float,
    dp_shift: float,
    dp_thresh: float,
    alpha: float,
    beta: float,
    max_genes: int,
    weight_coef_fit: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This performs omp on every pixel, the stopping criterion is that the dot_product_score when selecting the next gene
    to add exceeds dp_thresh or the number of genes added to the pixel exceeds max_genes.

    Args:
        pixel_colors: `float [n_pixels x n_rounds x n_channels]`.
            Pixel colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        background_shift: When fitting background,
            this is applied to weighting of each background vector to limit boost of weak pixels.
        dp_shift: When finding `dot_product_score` between residual `pixel_colors` and `bled_codes`,
            this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.
        dp_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added at each iteration.
        alpha: Used for `fitting_standard_deviation`, by how much to increase variance as genes added.
        beta: Used for `fitting_standard_deviation`, the variance with no genes added (`coef=0`) is `beta**2`.
        max_genes: Maximum number of genes that can be added to a pixel i.e. number of iterations of OMP.
        weight_coef_fit: If False, coefs are found through normal least squares fitting.
            If True, coefs are found through weighted least squares fitting using 1/sigma as the weight factor.

    Returns:
        - gene_coefs - `float32 [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        - background_coefs - `float32 [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.

    Notes:
        - Background vectors are fitted first and then not updated again.
        - All variables used in the for loop over OMP iterations is kept in GPU memory. They are then moved to the CPU
            device when all iterations are complete.
    """
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    n_pixels = pixel_colors.shape[0]
    pixel_colors = pixel_colors
    bled_codes = bled_codes
    torch.manual_seed(0)
    check_spot = torch.randint(0, n_pixels, size=(1,))[0].item()
    diff_to_int = torch.round(pixel_colors[check_spot]).to(int) - pixel_colors[check_spot]
    if torch.abs(diff_to_int).max() == 0:
        log.error(
            ValueError(
                f"pixel_coefs should be found using normalised pixel_colors."
                f"\nBut for pixel {check_spot}, pixel_colors given are integers indicating they are "
                f"the raw intensities."
            )
        )

    n_genes, n_rounds, n_channels = bled_codes.shape
    if not utils.errors.check_shape(pixel_colors, [n_pixels, n_rounds, n_channels]):
        log.error(utils.errors.ShapeError("pixel_colors", pixel_colors.shape, (n_pixels, n_rounds, n_channels)))
    no_verbose = n_pixels < 1000  # show progress bar with more than 1000 pixels.

    # Fit background and override initial pixel_colors
    gene_coefs = torch.zeros((n_pixels, n_genes), dtype=torch.float32)  # coefs of all genes and background
    pixel_colors, background_coefs, background_codes = call_spots.fit_background(pixel_colors, background_shift)

    background_genes = torch.arange(n_genes, n_genes + n_channels)

    # colors and codes for get_best_gene function
    # Includes background as if background is the best gene, iteration ends.
    # uses residual color as used to find next gene to add.
    bled_codes = bled_codes.reshape((n_genes, -1))
    all_codes = torch.concatenate((bled_codes, background_codes.reshape(n_channels, -1)))
    bled_codes = bled_codes.T

    # colors and codes for fit_coefs function (No background as this is not updated again).
    # always uses post background color as coefficients for all genes re-estimated at each iteration.
    pixel_colors = pixel_colors.reshape((n_pixels, -1)).to(cuda)

    all_codes = all_codes.to(cuda)
    background_coefs = background_coefs.to(cuda)
    background_genes = background_genes.to(cuda)
    bled_codes = bled_codes.to(cuda)
    gene_coefs = gene_coefs.to(cuda)

    continue_pixels = torch.arange(n_pixels, device=cuda)
    log.debug("Finding OMP coefficients started")
    with tqdm.tqdm(total=max_genes, disable=no_verbose, desc="Finding OMP coefficients for each pixel") as pbar:
        for i in range(max_genes):
            if i == 0:
                # Background coefs don't change, hence contribution to variance won't either.
                added_genes, pass_score_thresh, background_variance = get_best_gene_first_iter(
                    pixel_colors, all_codes, background_coefs, dp_shift, dp_thresh, alpha, beta, background_genes
                )
                inverse_var = 1 / background_variance
                pixel_colors = pixel_colors.T
            else:
                # only continue with pixels for which dot product score exceeds threshold
                i_added_genes, pass_score_thresh, inverse_var = get_best_gene(
                    residual_pixel_colors,
                    all_codes,
                    i_coefs,
                    added_genes,
                    dp_shift,
                    dp_thresh,
                    alpha,
                    background_genes,
                    background_variance,
                )

                # For pixels with at least one non-zero coef, add to final gene_coefs when fail the thresholding.
                fail_score_thresh = torch.logical_not(pass_score_thresh)
                # gene_coefs[torch.asarray(continue_pixels[fail_score_thresh])] = torch.asarray(i_coefs[fail_score_thresh])
                gene_coefs[
                    torch.asarray(continue_pixels[fail_score_thresh], device=cuda)[:, None],
                    torch.asarray(added_genes[fail_score_thresh], device=cuda),
                ] = torch.asarray(i_coefs[fail_score_thresh], device=cuda)

            continue_pixels = continue_pixels[pass_score_thresh]
            n_continue = len(continue_pixels)
            pbar.set_postfix({"n_pixels": n_continue, "gpu": "true"})
            if n_continue == 0:
                break
            if i == 0:
                added_genes = added_genes[pass_score_thresh, None]
            else:
                added_genes = torch.hstack((added_genes[pass_score_thresh], i_added_genes[pass_score_thresh, None]))
            pixel_colors = pixel_colors[:, pass_score_thresh]
            background_variance = background_variance[pass_score_thresh]
            inverse_var = inverse_var[pass_score_thresh]

            # Maybe add different fit_coefs for i==0 i.e. can do multiple pixels at once for same gene added.
            if weight_coef_fit:
                residual_pixel_colors, i_coefs = fit_coefs_weight(
                    bled_codes, pixel_colors, added_genes, torch.sqrt(inverse_var)
                )
            else:
                residual_pixel_colors, i_coefs = fit_coefs(bled_codes, pixel_colors, added_genes)

            if i == max_genes - 1:
                # Add pixels to final gene_coefs when reach end of iteration.
                gene_coefs[
                    torch.asarray(continue_pixels, device=cuda)[:, None], torch.asarray(added_genes, device=cuda)
                ] = torch.asarray(i_coefs, device=cuda)
            pbar.update()
    log.debug("Finding OMP coefficients complete")

    gene_coefs_cpu = gene_coefs.type(torch.float32).to(cpu)
    background_coefs_cpu = background_coefs.type(torch.float32).to(cpu)
    del (
        all_codes,
        background_coefs,
        background_genes,
        bled_codes,
        gene_coefs,
        added_genes,
        pass_score_thresh,
        background_variance,
        i_added_genes,
        inverse_var,
        fail_score_thresh,
        i_coefs,
        residual_pixel_colors,
    )

    return gene_coefs_cpu, background_coefs_cpu


def get_pixel_coefs_yxz(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_extract: NotebookPage,
    nbp_filter: NotebookPage,
    config: dict,
    tile: int,
    use_z: List[int],
    z_chunk_size: int,
    n_genes: int,
    transform: Union[torch.Tensor, torch.Tensor],
    color_norm_factor: Union[torch.Tensor, torch.Tensor],
    initial_intensity_thresh: float,
    bled_codes: Union[torch.Tensor, torch.Tensor],
    dp_norm_shift: Union[int, float],
) -> Tuple[np.ndarray, Any]:
    """
    Get each pixel OMP coefficients for a particular tile.

    Args:
        nbp_basic (NotebookPage): notebook page for 'basic_info'.
        nbp_file (NotebookPage): notebook page for 'file_names'.
        nbp_extract (NotebookPage): notebook page for 'extract'.
        config (dict): config settings for 'omp'.
        tile (int): tile index.
        use_z (list of int): list of z planes to calculate on.
        z_chunk_size (int): size of each z chunk.
        n_genes (int): the number of genes.
        transform (`[n_tiles x n_rounds x n_channels x 4 x 3] ndarray[float]`): `transform[t, r, c]` is the affine
            transform to get from tile `t`, `ref_round`, `ref_channel` to tile `t`, round `r`, channel `c`.
        color_norm_factor (`[n_rounds x n_channels] ndarray[float]`): Normalisation factors to divide colours by to
            equalise channel intensities.
        initial_intensity_thresh (float): pixel intensity threshold, only keep ones above the threshold to save memory
            and storage space.
        bled_codes (`[n_genes x n_rounds x n_channels] ndarray[float]`): bled codes.
        dp_norm_shift (int or float): when finding `dot_product_score` between residual `pixel_colors` and
            `bled_codes`, this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.

    Returns:
        - (`[n_pixels x 3] ndarray[int]`): `pixel_yxz_t` is the y, x and z pixel positions of the gene coefficients
            found.
        - (`[n_pixels x n_genes]`): `pixel_coefs_t` contains the gene coefficients for each pixel.
    """
    # Turn off memory fragmentation on GPU. This stops pytorch from crashing because of large memory allocations.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    pixel_yxz_t = np.zeros((0, 3), dtype=np.int16)
    pixel_coefs_t = scipy.sparse.csr_matrix(np.zeros((0, n_genes), dtype=np.float32))

    z_chunks = len(use_z) // z_chunk_size + 1
    for z_chunk in range(z_chunks):
        log.info(f"z_chunk {z_chunk + 1}/{z_chunks}")
        # While iterating through tiles, only save info for rounds/channels using
        # - add all rounds/channels back in later. This returns colors in use_rounds/channels only and no invalid.
        pixel_yxz_tz, pixel_colors_tz = base.get_pixel_colours(
            nbp_basic,
            nbp_file,
            nbp_extract,
            nbp_filter,
            int(tile),
            z_chunk,
            z_chunk_size,
            np.asarray(transform),
            np.asarray(color_norm_factor),
        )
        pixel_yxz_tz = torch.from_numpy(pixel_yxz_tz).type(torch.int16)
        pixel_colors_tz = torch.from_numpy(pixel_colors_tz).type(torch.float32)

        # Only keep pixels with significant absolute intensity to save memory.
        # absolute because important to find negative coefficients as well.
        pixel_intensity_tz = torch.from_numpy(call_spots.get_spot_intensity(torch.abs(pixel_colors_tz)))
        keep = pixel_intensity_tz > initial_intensity_thresh
        if not keep.any():
            continue
        pixel_colors_tz = pixel_colors_tz[keep]
        pixel_yxz_tz = pixel_yxz_tz[keep]
        del pixel_intensity_tz, keep

        bled_codes = torch.asarray(bled_codes, dtype=torch.float32)
        pixel_coefs_tz = get_all_coefs(
            pixel_colors_tz,
            bled_codes,
            0,
            dp_norm_shift,
            config["dp_thresh"],
            config["alpha"],
            config["beta"],
            config["max_genes"],
            config["weight_coef_fit"],
        )[0]
        with torch.no_grad():
            torch.cuda.empty_cache()
        pixel_coefs_tz = torch.asarray(pixel_coefs_tz, dtype=torch.float32)
        del pixel_colors_tz
        # Only keep pixels for which at least one gene has non-zero coefficient.
        keep = (torch.abs(pixel_coefs_tz).max(dim=1)[0] > 0).nonzero(as_tuple=True)[0]  # nonzero as is sparse matrix.
        if len(keep) == 0:
            continue
        pixel_yxz_t = np.append(pixel_yxz_t, np.asarray(pixel_yxz_tz[keep]), axis=0)
        pixel_coefs_t = scipy.sparse.vstack(
            (pixel_coefs_t, scipy.sparse.csr_matrix(np.asarray(pixel_coefs_tz[keep], np.float32)))
        )
        del pixel_yxz_tz, pixel_coefs_tz, keep

    return pixel_yxz_t, pixel_coefs_t
