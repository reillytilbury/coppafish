import os
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np
import torch

from ... import spot_colors
from ...call_spots import background_pytorch
from ...omp import coefs_torch, scores_torch
from ...setup import Notebook


def get_spot_position_and_tile(nb: Notebook, spot_no: int, method: str) -> Tuple[np.ndarray, int]:
    if method in ("anchor", "prob"):
        local_yxz = nb.ref_spots.local_yxz[spot_no]
        tile = nb.ref_spots.tile[spot_no]
    elif method == "omp":
        local_yxz = nb.omp.local_yxz[spot_no]
        tile = nb.omp.tile[spot_no]
    else:
        raise ValueError(f"Unknown gene calling method: {method}")
    return local_yxz, int(tile)


class ViewOMPImage:
    def __init__(
        self,
        nb: Notebook,
        spot_no: int,
        method: str,
        im_size: int = 8,
        z_planes: Tuple[int] = (-2, -1, 0, 1, 2),
        init_select_gene: int = None,
    ) -> None:
        """
        Display omp coefficients of all genes in around the local neighbourhood of spot.

        Args:
            nb (Notebook): Notebook containing experiment details.
            spot_no (int-like or none): Spot index to be plotted.
            method (str): gene calling method.
            im_size (int): number of pixels out from the central pixel to plot to create the square images.
            z_planes (tuple of int): z planes to show. 0 is the central z plane.
            init_select_gene (int): gene number to display initially. Default: the highest scoring gene.
        """
        assert type(nb) is Notebook
        if spot_no is None:
            return
        assert type(int(spot_no)) is int
        assert type(method) is str
        assert type(im_size) is int
        assert im_size >= 0
        assert type(z_planes) is tuple
        assert len(z_planes) > 2
        tile_dir = nb.file_names.tile_dir
        assert os.path.isdir(tile_dir), f"Viewing coefficients requires access to images expected at {tile_dir}"
        assert init_select_gene is None or type(init_select_gene) is int

        plt.style.use("dark_background")

        local_yxz, tile = get_spot_position_and_tile(nb, spot_no, method)
        assert local_yxz.shape == (3,)

        config = nb.init_config["omp"]

        coord_min = (local_yxz - im_size).tolist()
        coord_min[2] = local_yxz[2].item() + min(z_planes)
        coord_max = (local_yxz + im_size + 1).tolist()
        coord_max[2] = local_yxz[2].item() + max(z_planes) + 1
        yxz = [np.arange(coord_min[i], coord_max[i]) for i in range(3)]
        yxz = np.array(np.meshgrid(*[np.arange(coord_min[i], coord_max[i]) for i in range(3)])).reshape((3, -1)).T

        spot_shape_yxz = tuple([coord_max[i] - coord_min[i] for i in range(3)])
        n_rounds_use, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        image_colours = np.zeros(spot_shape_yxz + (n_rounds_use, n_channels_use), dtype=np.float32)
        for i, r in enumerate(nb.basic_info.use_rounds):
            image_colours[:, :, :, i] = spot_colors.base.get_spot_colours_new(
                nb.basic_info,
                nb.file_names,
                nb.extract,
                nb.register,
                nb.register_debug,
                int(tile),
                r,
                yxz=yxz,
                registration_type="flow_and_icp",
            ).T.reshape((spot_shape_yxz + (n_channels_use,)))
        image_colours = torch.asarray(image_colours, dtype=torch.float32)
        colour_norm_factor = np.array(nb.call_spots.color_norm_factor, dtype=np.float32)
        colour_norm_factor = colour_norm_factor[
            np.ix_(range(colour_norm_factor.shape[0]), nb.basic_info.use_rounds, nb.basic_info.use_channels)
        ]
        colour_norm_factor = torch.asarray(colour_norm_factor).float()
        bled_codes_ge = nb.call_spots.bled_codes_ge
        n_genes = bled_codes_ge.shape[0]
        bled_codes_ge = bled_codes_ge[np.ix_(range(n_genes), nb.basic_info.use_rounds, nb.basic_info.use_channels)]
        bled_codes_ge = torch.asarray(bled_codes_ge.astype(np.float32))

        image_colours = image_colours.reshape((-1, n_rounds_use, n_channels_use))
        bled_codes_ge = bled_codes_ge.reshape((n_genes, n_rounds_use * n_channels_use))

        image_colours /= colour_norm_factor[[tile]]
        image_colours, bg_coefficients, bg_codes = background_pytorch.fit_background(image_colours)
        image_colours = image_colours.reshape((-1, n_rounds_use * n_channels_use))
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
        ).toarray()
        coefficient_image = torch.asarray(coefficient_image).T.reshape(
            (len(nb.call_spots.gene_names),) + spot_shape_yxz
        )

        self.scores = []
        for g in range(coefficient_image.shape[0]):
            self.scores.append(
                scores_torch.score_coefficient_image(
                    coefficient_image[g],
                    (torch.asarray(spot_shape_yxz) // 2)[np.newaxis],
                    torch.asarray(nb.omp.spot),
                    torch.asarray(nb.omp.mean_spot),
                    config["high_coef_bias"],
                ).item()
            )

        # Of shape (n_genes, n_pixels)
        if init_select_gene is None:
            self.selected_gene = np.argmax(self.scores).item()
        else:
            self.selected_gene = init_select_gene
        self.gene_names = nb.call_spots.gene_names
        self.z_planes = z_planes
        self.coefficient_image = coefficient_image
        self.iteration_count_image = (~np.isclose(coefficient_image.numpy(), 0)).astype(int).sum(0)
        self.mid_z = -min(self.z_planes)
        self.function_coefficients = False
        self.show_iteration_counts = False
        self.high_coef_bias = config["high_coef_bias"]
        self.draw_canvas()
        plt.show()

    def draw_canvas(self) -> None:
        self.fig, self.axes = plt.subplots(
            nrows=2,
            ncols=len(self.z_planes) + 1,
            squeeze=False,
            gridspec_kw={"width_ratios": [5] * len(self.z_planes) + [1] * 1, "height_ratios": [6, 1]},
            layout="tight",
        )
        self.fig.subplots_adjust(bottom=0.25)
        ax_function_coefs = self.axes[1, 1]
        # Keep widgets in self otherwise they will get garbage collected and not respond to clicks anymore.
        self.function_coefs_button = CheckButtons(
            ax_function_coefs,
            ["Non-linear function"],
            actives=[self.function_coefficients],
            frame_props={"edgecolor": "white", "facecolor": "white"},
            check_props={"facecolor": "black"},
        )
        self.function_coefs_button.on_clicked(self.function_gene_coefficients_updated)
        ax_slider: plt.Axes = self.axes[1, 0]
        self.gene_slider = Slider(
            ax_slider,
            label="Gene",
            valmin=0,
            valmax=self.coefficient_image.shape[0] - 1,
            valstep=1,
            valinit=self.selected_gene,
        )
        self.gene_slider.active = True
        self.gene_slider.on_changed(self.gene_selected_updated)
        ax_iteration_count = self.axes[1, 2]
        self.show_iteration_count_button = CheckButtons(
            ax_iteration_count,
            ["Show iteration counts"],
            actives=[self.show_iteration_counts],
            frame_props={"edgecolor": "white", "facecolor": "white"},
            check_props={"facecolor": "black"},
        )
        self.show_iteration_count_button.on_clicked(self.show_iteration_count_changed)
        self.draw_data()

    def draw_data(self) -> None:
        cmap = mpl.cm.viridis
        title = f"Gene {self.selected_gene} {self.gene_names[self.selected_gene]}"

        if self.show_iteration_counts:
            norm = mpl.colors.Normalize(vmin=0, vmax=self.iteration_count_image.max())
            image_data = self.iteration_count_image
            title += "\nIteration Count"
        else:
            image_data = self.coefficient_image.detach().clone()[self.selected_gene]
            title += f" Score: {round(self.scores[self.selected_gene], 3)}\n"
            title += "OMP Coefficients"
            if self.function_coefficients:
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                image_data = coefs_torch.non_linear_function_coefficients(image_data, self.high_coef_bias)
            else:
                abs_max = np.abs(self.coefficient_image).max()
                norm = mpl.colors.Normalize(vmin=min(0, self.coefficient_image.min()), vmax=abs_max)
            image_data = image_data.numpy()

        for ax in self.axes[0]:
            ax.clear()
        all_spines = ("top", "bottom", "left", "right")
        for ax in self.axes[1]:
            for spine in all_spines:
                ax.spines[spine].set_visible(False)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
        self.fig.suptitle(title)
        self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.axes[0, -1],
            orientation="vertical",
            label="",
        )
        for i, z_plane in enumerate(self.z_planes):
            ax: plt.Axes = self.axes[0, i]
            ax.clear()
            ax.imshow(image_data[:, :, self.mid_z + z_plane], cmap=cmap, norm=norm)
            ax_title = "Central plane"
            if z_plane < 0:
                ax_title = f"- {abs(z_plane)}"
            if z_plane > 0:
                ax_title = f"+ {abs(z_plane)}"
            ax.set_title(ax_title)
        self.gene_slider.active = True
        plt.draw()

    def function_gene_coefficients_updated(self, _) -> None:
        self.function_coefficients = self.function_coefs_button.get_status()[0]
        self.draw_data()

    def show_iteration_count_changed(self, _) -> None:
        self.show_iteration_counts = self.show_iteration_count_button.get_status()[0]
        self.draw_data()

    def gene_selected_updated(self, _) -> None:
        self.selected_gene = int(self.gene_slider.val)
        self.draw_data()


class ViewOMPPixelCoefficients:
    def __init__(self, nb: Notebook, spot_no: int, method: str, tile: int = None, local_yxz: np.ndarray = None) -> None:
        """
        Show the OMP coefficients for one pixel position over each OMP iteration.

        Args:
            nb (Notebook): the notebook.
            spot_no (int): the spot index.
            method (str): the method that the spot was found on. Can be 'anchor', 'prob' or 'omp'.
            tile (int): tile of the pixel.
            local_yxz (`(3) ndarray[int]`): position relative to tile to view. If tile and local_yxz are both given,
                then spot_no and method are ignored.
        """
        assert type(nb) is Notebook
        assert type(spot_no) is int
        assert type(method) is str
        assert method in ("anchor", "prob", "omp")
        assert tile is None or type(tile) is int
        assert local_yxz is None or type(local_yxz) is np.ndarray

        if tile is None or local_yxz is None:
            local_yxz, tile = get_spot_position_and_tile(nb, spot_no, method)

        n_rounds_use, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        image_colours = np.zeros((1, n_rounds_use, n_channels_use), dtype=np.float32)
        for i, r in enumerate(nb.basic_info.use_rounds):
            image_colours[:, i] = spot_colors.base.get_spot_colours_new(
                nb.basic_info,
                nb.file_names,
                nb.extract,
                nb.register,
                nb.register_debug,
                int(tile),
                r,
                yxz=local_yxz[np.newaxis],
                registration_type="flow_and_icp",
            ).T[np.newaxis]
        image_colours = torch.asarray(image_colours, dtype=torch.float32)
        colour_norm_factor = np.array(nb.call_spots.color_norm_factor, dtype=np.float32)
        colour_norm_factor = colour_norm_factor[
            np.ix_(range(colour_norm_factor.shape[0]), nb.basic_info.use_rounds, nb.basic_info.use_channels)
        ]
        colour_norm_factor = torch.asarray(colour_norm_factor).float()
        bled_codes_ge = nb.call_spots.bled_codes_ge
        n_genes = bled_codes_ge.shape[0]
        bled_codes_ge = bled_codes_ge[np.ix_(range(n_genes), nb.basic_info.use_rounds, nb.basic_info.use_channels)]
        bled_codes_ge = torch.asarray(bled_codes_ge.astype(np.float32))

        image_colours = image_colours.reshape((-1, n_rounds_use, n_channels_use))
        bled_codes_ge = bled_codes_ge.reshape((n_genes, n_rounds_use * n_channels_use))

        image_colours /= colour_norm_factor[[tile]]
        image_colours, bg_coefficients, bg_codes = background_pytorch.fit_background(image_colours)
        image_colours = image_colours.reshape((-1, n_rounds_use * n_channels_use))
        bg_codes = bg_codes.reshape((n_channels_use, n_rounds_use * n_channels_use))

        config = nb.init_config["omp"]
        # Get the maximum number of OMP gene assignments made and what genes.
        coefficients = coefs_torch.compute_omp_coefficients(
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
        ).toarray()[0]
        self.maximum_iterations = (~np.isclose(coefficients, 0)).sum()
        if self.maximum_iterations == 0:
            raise ValueError(f"The selected pixel has no OMP gene assignments to display")
        self.final_selected_genes = (~np.isclose(coefficients, 0)).nonzero()[0].tolist()
        self.coefficients = np.zeros((self.maximum_iterations, len(self.final_selected_genes)), dtype=np.float32)
        for i in range(self.maximum_iterations):
            self.coefficients[i] = coefs_torch.compute_omp_coefficients(
                image_colours,
                bled_codes_ge,
                maximum_iterations=(i + 1),
                background_coefficients=bg_coefficients,
                background_codes=bg_codes,
                dot_product_threshold=config["dp_thresh"],
                dot_product_norm_shift=0.0,
                weight_coefficient_fit=config["weight_coef_fit"],
                alpha=config["alpha"],
                beta=config["beta"],
                do_not_compute_on=None,
                force_cpu=config["force_cpu"],
            ).toarray()[0][self.final_selected_genes]
        self.local_yxz = local_yxz
        self.gene_names = nb.call_spots.gene_names
        self.show_iteration = self.maximum_iterations - 1
        self.draw_canvas()
        self.draw_data()
        plt.show()

    def draw_canvas(self) -> None:
        plt.style.use("dark_background")
        self.fig, self.axes = plt.subplots(2, 1, squeeze=False, gridspec_kw={"height_ratios": [7, 1]})
        self.fig.suptitle(f"OMP coefficients for pixel {tuple(self.local_yxz.tolist())}")
        ax_slider: plt.Axes = self.axes[1, 0]
        self.iteration_slider = Slider(
            ax_slider,
            label="Iteration",
            valmin=1,
            valmax=self.maximum_iterations,
            valstep=1,
            valinit=self.maximum_iterations,
        )
        self.iteration_slider.active = True
        self.iteration_slider.on_changed(self.show_iteration_changed)

    def draw_data(self) -> None:
        ax_plot: plt.Axes = self.axes[0, 0]
        ax_plot.clear()
        x_min, x_max = -0.5, self.maximum_iterations - 0.5
        ax_plot.set_xlim(x_min, x_max)
        abs_max = np.abs(self.coefficients).max()
        ax_plot.set_ylim(-abs_max - 0.5, abs_max + 0.5)
        ax_plot.hlines(0, x_min, x_max, colors="white", linewidths=1.0)
        ax_plot.bar(
            np.linspace(0, self.maximum_iterations, num=self.maximum_iterations, endpoint=False),
            self.coefficients[self.show_iteration],
            width=0.3,
            color="whitesmoke",
            edgecolor="dimgrey",
            linewidth=1.5,
            tick_label=[self.gene_names[i] for i in self.final_selected_genes],
        )
        plt.draw()

    def show_iteration_changed(self, _) -> None:
        self.show_iteration = (self.iteration_slider.val) - 1
        self.draw_data()
