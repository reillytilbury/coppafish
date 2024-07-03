import numpy as np
from coppafish.plot.call_spots import parameter_estimation


def test_all_viewers():
    n_genes, n_dyes, n_tiles, n_rounds, n_channels, n_spots = 10, 3, 5, 3, 4, 100
    free_bled_codes_tile_indep, free_bled_codes, bled_codes = (np.random.rand(n_genes, n_dyes, n_channels),
                                                               np.random.rand(n_genes, n_tiles, n_dyes, n_channels),
                                                               np.random.rand(n_genes, n_rounds, n_channels))
    gene_names, gene_codes = (np.array([f'gene_{i}' for i in range(n_genes)]),
                              np.random.randint(0, n_dyes, (n_genes, n_rounds)))
    tile_scale, rc_scale = np.random.rand(n_tiles, n_rounds, n_channels), np.random.rand(n_rounds, n_channels)
    n_spots = np.random.randint(0, 100, n_genes)
    d_max = [0, 1, 2, 0]
    # 1. view_free_and_constrained_bled_codes
    parameter_estimation.view_free_and_constrained_bled_codes(free_bled_codes_tile_indep=free_bled_codes_tile_indep,
                                                              bled_codes=bled_codes, gene_names=gene_names,
                                                              rc_scale=rc_scale, n_spots=n_spots, show=False)
    # 2. view_tile_bled_codes
    parameter_estimation.view_tile_bled_codes(free_bled_codes=free_bled_codes,
                                              free_bled_codes_tile_indep=free_bled_codes_tile_indep,
                                              gene_names=gene_names, use_tiles=np.arange(n_tiles),
                                              gene=np.random.randint(0, n_genes), show=False)
    # 3. view_rc_scale_regression
    parameter_estimation.view_rc_scale_regression(rc_scale=rc_scale, gene_codes=gene_codes, d_max=d_max,
                                                  target_values=np.random.rand(n_channels),
                                                  free_bled_codes_tile_indep=free_bled_codes_tile_indep,
                                                  n_spots=n_spots, use_channels=np.arange(n_channels), show=False)

    # 4. view_tile_scale_regression
    parameter_estimation.view_tile_scale_regression(tile_scale=tile_scale, gene_codes=gene_codes, d_max=d_max,
                                                    target_bled_codes=free_bled_codes_tile_indep * rc_scale[None, :, :],
                                                    free_bled_codes=free_bled_codes, n_spots=n_spots, t=0,
                                                    use_channels=np.arange(n_channels), show=False)

    # 5. view_scale_factors
    parameter_estimation.view_scale_factors(tile_scale=tile_scale, rc_scale=rc_scale, use_tiles=np.arange(n_tiles),
                                            use_rounds=np.arange(n_rounds), use_channels=np.arange(n_channels),
                                            show=False)
