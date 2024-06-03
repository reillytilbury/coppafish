import itertools
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ... import register
from ...call_spots import background_pytorch
from ...omp import coefs_torch
from ...setup import Notebook


class View_OMP_Coefficients:
    def __init__(
        self, nb: Notebook, spot_no: int, method: str, im_size: int = 8, z_planes: Tuple[int] = (-2, 0, 2)
    ) -> None:
        """
        Display omp coefficients of all genes in neighbourhood of spot in three z planes.

        Args:
            nb (Notebook): Notebook containing experiment details.
            spot_no (int): Spot index to be plotted.
            method (str): gene calling method.
            im_size (int): number of pixels out from the central pixel to plot to create the square images.
            z_planes (tuple of int): z planes to show. 0 is the central z plane.
        """
        assert type(nb) is Notebook
        assert type(spot_no) is int
        assert type(method) is str
        assert type(im_size) is int
        assert im_size >= 0
        assert type(z_planes) is tuple
        assert len(z_planes) > 0
        tile_dir = nb.file_names.tile_dir
        assert os.path.isdir(tile_dir), f"Viewing coefficients requires access to images expected at {tile_dir}"

        plt.style.use("dark_background")
        local_yxz: np.ndarray = None
        tile: int = None
        if method in ("anchor", "prob"):
            local_yxz = nb.ref_spots.local_yxz[spot_no]
            tile = nb.ref_spots.tile[spot_no]
        elif method == "omp":
            local_yxz = nb.omp.local_yxz[spot_no]
            tile = nb.omp.tile[spot_no]
        else:
            raise ValueError(f"Unknown gene calling method: {method}")
        assert local_yxz.shape == (3,)

        config = nb.init_config["omp"]

        coord_min = local_yxz - im_size
        coord_min[2] = local_yxz[2] + min(z_planes)
        coord_max = local_yxz + im_size + 1
        coord_max[2] = local_yxz[2] + max(z_planes)
        yxz = []
        for i in range(3):
            yxz.append([coord_min[i], coord_max[i]])

        spot_shape = tuple([coord_max[i] - coord_min[i] for i in range(3)])
        n_rounds_use, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        image_colours = np.zeros(spot_shape + (n_rounds_use, n_channels_use), dtype=np.float32)
        for r, c in itertools.product(nb.basic_info.use_rounds, nb.basic_info.use_channels):
            image_colours[:, :, :, r, c] = register.preprocessing.load_transformed_image(
                nb.basic_info,
                nb.file_names,
                nb.extract,
                nb.register,
                nb.register_debug,
                tile,
                r,
                c,
                yxz,
                reg_type="flow_icp",
            )
        image_colours = torch.asarray(image_colours, dtype=torch.float32)
        bled_codes_ge = nb.call_spots.bled_codes_ge
        n_genes = bled_codes_ge.shape[0]
        bled_codes_ge = bled_codes_ge[np.ix_(range(n_genes), nb.basic_info.use_rounds, nb.basic_info.use_channels)]
        bled_codes_ge = torch.asarray(bled_codes_ge.astype(np.float32))

        image_colours = image_colours.reshape((-1, n_rounds_use, n_channels_use))
        bled_codes_ge = bled_codes_ge.reshape((n_genes, n_rounds_use * n_channels_use))

        image_colours, bg_coefficients, bg_codes = background_pytorch.fit_background(image_colours)
        bg_codes = bg_codes.reshape((n_channels_use, n_rounds_use * n_channels_use))

        coefficient_image = coefs_torch.compute_omp_coefficients(
            image_colours,
            bled_codes_ge,
            maximum_iterations=config["max_genes"],
            background_coefficients=bg_coefficients,
            background_codes=bg_codes,
            dot_product_threshold=config["dp_thresh"],
            dot_product_norm_shift=0.0,
            weight_coefficient_fit=config["weight_coef_fit"],
            alpha=config["alpha"],
            beta=config["beta"],
            do_not_compute_on=None,
            force_cpu=config["force_cpu"],
        )
        # Of shape (n_genes, n_pixels)
        coefficient_image: np.ndarray = coefficient_image.toarray()
        coefficient_image = coefficient_image.reshape((-1, spot_shape))
        mid_z = spot_shape[2] // 2

        fig, axes = plt.subplots(nrows=1, ncols=len(z_planes), squeeze=False)
        for i, z_plane in enumerate(z_planes):
            ax: plt.Axes = axes[0, i]
            ax.imshow(coefficient_image[0, :, :, mid_z + z_plane])
            ax_title = "Central plane"
            if z_plane < 0:
                ax_title = f"- {abs(z_plane)}"
            if z_plane > 0:
                ax_title = f"+ {abs(z_plane)}"
            ax.set_title(ax_title)
        fig.tight_layout()
        plt.show()
