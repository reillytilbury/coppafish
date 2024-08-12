from typing import Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np
import torch

from ... import spot_colors
from ...call_spots import background_pytorch
from ...omp import coefs, scores_torch
from ...omp import base as omp_base
from ...setup import Notebook


def get_spot_position_and_tile(nb: Notebook, spot_no: int, method: str) -> Tuple[np.ndarray, int]:
    if method in ("anchor", "prob"):
        local_yxz = nb.ref_spots.local_yxz[spot_no]
        tile = nb.ref_spots.tile[spot_no]
    elif method == "omp":
        all_local_yxz, all_tile = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)
        local_yxz = all_local_yxz[spot_no]
        tile = all_tile[spot_no].item()
    else:
        raise ValueError(f"Unknown gene calling method: {method}")
    return local_yxz, int(tile)


class ViewOMPImage:
    def __init__(
        self,
        nb: Notebook,
        spot_no: Union[int, None],
        method: str,
        im_size: int = 8,
        z_planes: Tuple[int] = (-2, -1, 0, 1, 2),
        init_select_gene: Union[int, None] = None,
    ) -> None:
        """
        Display omp coefficients of all genes around the local neighbourhood of a spot.

        Args:
            - nb (Notebook): Notebook containing experiment details.
            - spot_no (int-like or none): Spot index to be plotted.
            - method (str): gene calling method.
            - im_size (int): number of pixels out from the central pixel to plot to create the square images.
            - z_planes (tuple of int): z planes to show. 0 is the central z plane.
            - init_select_gene (int): gene number to display initially. Default: the highest scoring gene.
        """
        assert type(nb) is Notebook
        if spot_no is None:
            return
        assert type(int(spot_no)) is int
        assert type(method) is str
        assert type(im_size) is int
        assert im_size >= 0
        assert type(z_planes) is tuple
        assert init_select_gene is None or type(init_select_gene) is int

        plt.style.use("dark_background")

        local_yxz, tile = get_spot_position_and_tile(nb, spot_no, method)
        assert local_yxz.shape == (3,)

        config = nb.omp.associated_configs["omp"]

        coord_min = (local_yxz - im_size).tolist()
        coord_min[2] = local_yxz[2].item() + min(z_planes)
        coord_max = (local_yxz + im_size + 1).tolist()
        coord_max[2] = local_yxz[2].item() + max(z_planes) + 1
        yxz = [np.arange(coord_min[i], coord_max[i]) for i in range(3)]
        yxz = np.array(np.meshgrid(*[np.arange(coord_min[i], coord_max[i]) for i in range(3)])).reshape((3, -1)).T

        spot_shape_yxz = tuple([coord_max[i] - coord_min[i] for i in range(3)])
        central_yxz = tuple(torch.asarray(spot_shape_yxz)[np.newaxis].T.int() // 2)
        n_rounds_use, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        image_colours = np.zeros(spot_shape_yxz + (n_rounds_use, n_channels_use), dtype=np.float32)
        for i, r in enumerate(nb.basic_info.use_rounds):
            image_colours[:, :, :, i] = spot_colors.base.get_spot_colours_new(
                nb.filter.images,
                nb.register.flow,
                nb.register.icp_correction,
                nb.register_debug.channel_correction,
                nb.basic_info.use_channels,
                nb.basic_info.dapi_channel,
                int(tile),
                r,
                yxz=yxz,
                registration_type="flow_and_icp",
            ).T.reshape((spot_shape_yxz + (n_channels_use,)))
        assert not np.allclose(image_colours, 0)
        image_colours = image_colours.reshape((-1, n_rounds_use, n_channels_use))
        bled_codes = nb.call_spots.bled_codes.astype(np.float32)
        n_genes = bled_codes.shape[0]
        assert (~np.isnan(bled_codes)).all(), "bled codes cannot contain nan values"
        assert np.allclose(np.linalg.norm(bled_codes, axis=(1, 2)), 1), "bled codes must be L2 normalised"
        coefficient_image = coefs.compute_omp_coefficients(
            pixel_colours=image_colours,
            bled_codes=bled_codes,
            background_codes=np.eye(n_channels_use)[:, None, :].repeat(n_rounds_use, axis=1),
            colour_norm_factor=nb.call_spots.colour_norm_factor[[tile]].astype(np.float32),
            maximum_iterations=config["max_genes"],
            dot_product_threshold=config["dp_thresh"],
            normalisation_shift=config["lambda_d"],
            pixel_subset_count=config["subset_pixels"],
        )
        coefficient_image = coefficient_image.toarray()
        coefficient_image = torch.asarray(coefficient_image).T.reshape(
            (len(nb.call_spots.gene_names),) + spot_shape_yxz
        )

        self.scores = []
        for g in range(coefficient_image.shape[0]):
            self.scores.append(
                scores_torch.score_coefficient_image(
                    coefficient_image[[g]], torch.asarray(nb.omp.spot), torch.asarray(nb.omp.mean_spot)
                )[0][central_yxz].item()
            )
        self.scores = np.array(self.scores, np.float32)

        self.coefficient_image: np.ndarray = coefficient_image.numpy()

        central_pixel = np.array(self.coefficient_image.shape[1:]) // 2
        central_pixels = np.ix_(range(n_genes), [central_pixel[0]], [central_pixel[1]], [central_pixel[2]])
        gene_is_selectable = ~np.isclose(self.coefficient_image[central_pixels].ravel(), 0)
        if init_select_gene is not None:
            gene_is_selectable[init_select_gene] = True
        assert gene_is_selectable.ndim == 1

        self.gene_names = nb.call_spots.gene_names
        self.z_planes = z_planes
        self.selectable_genes = np.where(gene_is_selectable)[0]
        if init_select_gene is None:
            self.selected_gene = self.selectable_genes[np.argmax(self.scores[self.selectable_genes])].item()
        else:
            self.selected_gene = init_select_gene
        self.iteration_count_image = (~np.isclose(self.coefficient_image, 0)).astype(int).sum(0)
        self.mid_z = -min(self.z_planes)
        self.function_coefficients = False
        self.show_iteration_counts = False
        self.draw_canvas()
        plt.show()

    def draw_canvas(self) -> None:
        self.fig, self.axes = plt.subplots(
            nrows=2,
            ncols=len(self.z_planes) + 1,
            squeeze=False,
            gridspec_kw={"width_ratios": [5] * len(self.z_planes) + [1] * 1, "height_ratios": [6, 1]},
            layout="constrained",
            num="OMP Image",
        )
        # Keep widgets in self otherwise they will get garbage collected and not respond to clicks anymore.
        ax_slider: plt.Axes = self.axes[1, 0]
        self.gene_slider = Slider(
            ax_slider,
            label="Gene",
            valmin=self.selectable_genes.min(),
            valmax=self.selectable_genes.max(),
            valstep=self.selectable_genes,
            valinit=self.selected_gene,
        )
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

        if self.show_iteration_counts:
            cmap = mpl.cm.viridis
            image_data = self.iteration_count_image
            norm = mpl.colors.Normalize(vmin=0, vmax=self.iteration_count_image.max())
            title = "OMP Iteration Count"
        else:
            cmap = mpl.cm.PiYG
            image_data = self.coefficient_image[self.selected_gene]
            abs_max = np.abs(image_data).max()
            norm = mpl.colors.Normalize(vmin=-abs_max, vmax=abs_max)
            title = "OMP Coefficients\n"
            title += f"Gene {self.selected_gene} {self.gene_names[self.selected_gene]}\n"
            title += f" Score: {str(self.scores[self.selected_gene])[:4]}"

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
        self.gene_slider.active = not self.show_iteration_counts
        plt.draw()

    def show_iteration_count_changed(self, _) -> None:
        self.show_iteration_counts = self.show_iteration_count_button.get_status()[0]
        self.draw_data()

    def gene_selected_updated(self, _) -> None:
        self.selected_gene = int(self.gene_slider.val)
        self.draw_data()


class ViewOMPPixelColours:
    def __init__(self, nb: Notebook, spot_no: int, method: str) -> None:
        """
        Plot a pixel's true colour, the pixel's sum of gene's colour from OMP, and each OMP assigned gene colour after
        OMP coefficient weighting.

        Args:
            - nb (Notebook): the notebook including `omp`.
            - spot_no (int): the spot's index.
            - method (str): the spot's method, can be 'omp', 'prob', or 'anchor'.
        """
        assert type(nb) is Notebook
        assert type(spot_no) is int
        assert type(method) is str
        assert method in ("omp", "prob", "anchor")

        config = nb.omp.associated_configs["omp"]
        self.local_yxz, tile = get_spot_position_and_tile(nb, spot_no, method)

        n_rounds_use, n_channels_use = len(nb.basic_info.use_rounds), len(nb.basic_info.use_channels)
        image_colours = np.zeros((1, n_rounds_use, n_channels_use), dtype=np.float32)
        for i, r in enumerate(nb.basic_info.use_rounds):
            image_colours[:, i] = spot_colors.base.get_spot_colours_new(
                nb.filter.images,
                nb.register.flow,
                nb.register.icp_correction,
                nb.register_debug.channel_correction,
                nb.basic_info.use_channels,
                nb.basic_info.dapi_channel,
                int(tile),
                r,
                yxz=self.local_yxz[np.newaxis],
                registration_type="flow_and_icp",
            ).T[np.newaxis]
        image_colours[np.isnan(image_colours)] = 0
        assert not np.allclose(image_colours, 0)
        colour_norm_factor = np.array(nb.call_spots.colour_norm_factor, dtype=np.float32)
        colour_norm_factor = torch.asarray(colour_norm_factor).float()
        bled_codes = nb.call_spots.bled_codes
        assert (~np.isnan(bled_codes)).all(), "bled codes cannot contain nan values"
        assert np.allclose(np.linalg.norm(bled_codes, axis=(1, 2)), 1), "bled codes must be L2 normalised"

        # Get the maximum number of OMP gene assignments made and what genes.
        coefficients = coefs.compute_omp_coefficients(
            pixel_colours=image_colours,
            bled_codes=bled_codes,
            background_codes=np.eye(n_channels_use)[:, None, :].repeat(n_rounds_use, axis=1),
            colour_norm_factor=nb.call_spots.colour_norm_factor[[tile]].astype(np.float32),
            maximum_iterations=config["max_genes"],
            dot_product_threshold=config["dp_thresh"],
            normalisation_shift=config["lambda_d"],
            pixel_subset_count=config["subset_pixels"],
        ).toarray()[0]
        final_selected_genes = (~np.isclose(coefficients, 0)).nonzero()[0]
        self.n_assigned_genes: int = (~np.isclose(coefficients, 0)).sum().item()
        if self.n_assigned_genes == 0:
            raise ValueError(f"The selected pixel has no OMP gene assignments to display")
        # Show the zeroth iteration too with no genes assigned.
        self.coefficients = np.zeros((self.n_assigned_genes + 1, self.n_assigned_genes), dtype=np.float32)
        for i in range(1, self.n_assigned_genes + 1):
            self.coefficients[i] = coefs.compute_omp_coefficients(
                pixel_colours=image_colours,
                bled_codes=bled_codes,
                background_codes=np.eye(n_channels_use)[:, None, :].repeat(n_rounds_use, axis=1),
                colour_norm_factor=nb.call_spots.colour_norm_factor[[tile]].astype(np.float32),
                maximum_iterations=config["max_genes"],
                dot_product_threshold=config["dp_thresh"],
                normalisation_shift=config["lambda_d"],
                pixel_subset_count=config["subset_pixels"],
            ).toarray()[0, final_selected_genes]
        self.assigned_genes_names = nb.call_spots.gene_names[final_selected_genes]
        self.gene_bled_codes = bled_codes[final_selected_genes].reshape((-1, n_rounds_use, n_channels_use))
        self.gene_bled_codes = self.gene_bled_codes[np.newaxis].repeat(self.n_assigned_genes + 1, axis=0)
        self.gene_bled_codes *= self.coefficients[:, :, np.newaxis, np.newaxis]
        self.true_pixel_colour: np.ndarray = image_colours.reshape((n_rounds_use, n_channels_use))
        self.true_pixel_colour /= np.sqrt(np.square(self.true_pixel_colour).sum()) + config["lambda_d"]
        self.omp_final_colour = self.gene_bled_codes.sum(1)

        # Order genes based on their final coefficient strength.
        gene_order = np.argsort(np.abs(self.coefficients[-1]))[::-1]
        self.assigned_genes = np.array(range(self.n_assigned_genes))
        self.assigned_genes = self.assigned_genes[gene_order]
        self.assigned_genes_names = self.assigned_genes_names[gene_order]
        self.gene_bled_codes = self.gene_bled_codes[:, gene_order]

        self.selected_iteration = self.n_assigned_genes

        self.draw_canvas()
        self.draw_data()
        plt.show()

    def draw_canvas(self) -> None:
        # Axes for OMP assigned genes, 2 axes for the final OMP colour and the pixel's true colour, 1 for the colourbar.
        # 1 for the residual colour. 1 for the UI iteration slider.
        n_columns = max(self.n_assigned_genes, 5)
        n_rows = 2
        self.fig, self.axes = plt.subplots(n_rows, n_columns, squeeze=False, num="OMP Colour")
        self.axes = self.axes.ravel()
        self.fig.suptitle(f"OMP colour at pixel {tuple(self.local_yxz)}")

        abs_max_colour = np.abs(self.true_pixel_colour).max()
        abs_max_colour = np.max([abs_max_colour, np.abs(self.omp_final_colour).max()])
        abs_max_colour = np.max([abs_max_colour, np.abs(self.gene_bled_codes).max()])
        self.norm = mpl.colors.Normalize(vmin=-abs_max_colour, vmax=abs_max_colour)
        self.cmap = mpl.colormaps["bwr"]

        self.gene_images: list[plt.AxesImage] = []
        final_i = self.axes.size - 1
        for i, ax in enumerate(self.axes):
            ax: plt.Axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)
            ax.spines.left.set_visible(False)
            ax.spines.bottom.set_visible(False)
            ax_title = None
            shown_axes = False
            empty_data = np.zeros_like(self.true_pixel_colour.T)
            if i < self.assigned_genes.size:
                ax_title = f"{self.assigned_genes_names[i]}"
                self.gene_images.append(ax.imshow(empty_data, norm=self.norm, cmap=self.cmap))
                shown_axes = True
            elif i == final_i - 4:
                ax_title = f"True"
                ax.imshow(self.true_pixel_colour.T, norm=self.norm, cmap=self.cmap)
                shown_axes = True
            elif i == final_i - 3:
                ax_title = f"Fit"
                self.omp_final_im = ax.imshow(empty_data, norm=self.norm, cmap=self.cmap)
                shown_axes = True
            elif i == final_i - 2:
                ax_title = f"True - Fit"
                self.omp_residual_im = ax.imshow(empty_data, norm=self.norm, cmap=self.cmap)
                shown_axes = True
            elif i == final_i - 1:
                self.iteration_slider = Slider(
                    ax,
                    label="Iteration",
                    valmin=0,
                    valmax=self.n_assigned_genes,
                    valinit=self.selected_iteration + 1,
                    valstep=range(self.n_assigned_genes + 1),
                    orientation="horizontal",
                )
                self.iteration_slider.on_changed(self.iteration_slider_changed)
            elif i == final_i:
                self.fig.colorbar(
                    mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
                    cax=self.axes[final_i],
                    label="Pixel Intensity",
                )
            if shown_axes:
                # Y axis are rounds, x axis are channels.
                ax.set_xlabel(f"Round")
                ax.set_ylabel(f"Channel")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines.top.set_visible(True)
                ax.spines.right.set_visible(True)
                ax.spines.left.set_visible(True)
                ax.spines.bottom.set_visible(True)
            if ax_title is not None:
                ax.set_title(ax_title)

    def draw_data(self) -> None:
        for i, axes_im in enumerate(self.gene_images):
            i_gene_bled_code = self.gene_bled_codes[self.selected_iteration, i].T
            axes_im.set_data(self.gene_bled_codes[self.selected_iteration, i].T)

        final_omp_colour = self.omp_final_colour[self.selected_iteration]
        self.omp_residual_im.set_data(self.true_pixel_colour.T - final_omp_colour.T)
        self.omp_final_im.set_data(final_omp_colour.T)

        plt.draw()

    def iteration_slider_changed(self, _) -> None:
        new_selected_iteration = int(self.iteration_slider.val)
        if new_selected_iteration != self.selected_iteration:
            self.selected_iteration = new_selected_iteration
            self.draw_data()
