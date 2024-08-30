import numpy as np
from matplotlib import pyplot as plt
from ...setup import Notebook
from .spot_colours import ColorPlotBase


def ViewBleedMatrix(nb: Notebook):
    """
    Diagnostic to plot `bleed_matrix`.
    Args:
        nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
    """
    plt.style.use("dark_background")
    bleed_matrix_raw = nb.call_spots.bleed_matrix_raw
    bleed_matrix_initial = nb.call_spots.bleed_matrix_initial
    bleed_matrix = nb.call_spots.bleed_matrix

    # create figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(bleed_matrix_raw.T, cmap="viridis")
    ax[0].set_title("Raw Bleed Matrix")
    ax[1].imshow(bleed_matrix_initial.T, cmap="viridis")
    ax[1].set_title("Initial Bleed Matrix")
    ax[2].imshow(bleed_matrix.T, cmap="viridis")
    ax[2].set_title("Final Bleed Matrix")

    # add x and y labels and ticks
    dye_names = nb.basic_info.dye_names
    use_channels = nb.basic_info.use_channels
    for i in range(3):
        ax[i].set_xticks(ticks=np.arange(len(dye_names)), labels=dye_names, rotation=45)
        ax[i].set_yticks(ticks=np.arange(len(use_channels)), labels=use_channels)
        ax[i].set_xlabel("Dye")
        ax[i].set_ylabel("Channel")

    # add super title
    fig.suptitle("Bleed Matrix")

    plt.show()


class view_bled_codes(ColorPlotBase):
    def __init__(self, nb: Notebook):
        """
        Diagnostic to show `bled_codes` with and without `gene_efficiency` applied for all genes.
        Change gene by scrolling with mouse.

        Args:
            nb: Notebook containing experiment details. Must have run at least as far as `call_reference_spots`.
        """
        plt.style.use("dark_background")

        self.n_genes = nb.call_spots.bled_codes.shape[0]
        self.gene_names = nb.call_spots.gene_names
        self.use_rounds = nb.basic_info.use_rounds
        bled_codes = nb.call_spots.bled_codes
        bled_codes_initial = nb.call_spots.bleed_matrix[nb.call_spots.gene_codes]
        bled_codes_initial /= np.linalg.norm(bled_codes_initial, axis=(1, 2), keepdims=True)
        # move gene index to last so scroll over genes. After this shape is n_rounds x n_channels x n_genes
        bled_codes = np.moveaxis(bled_codes, 0, -1)
        bled_codes_initial = np.moveaxis(bled_codes_initial, 0, -1)
        # move channel index to first for plotting
        bled_codes = np.moveaxis(bled_codes, 0, 1)
        bled_codes_initial = np.moveaxis(bled_codes_initial, 0, 1)
        subplot_adjust = [0.07, 0.775, 0.095, 0.9]  # larger top adjust for super title
        super().__init__([bled_codes_initial, bled_codes], np.ones_like(bled_codes), subplot_adjust=subplot_adjust)
        n_channels = bled_codes.shape[1]
        for j in range(self.n_images):
            # plot a black horizontal line above every third channel
            for c in range(n_channels):
                if c % 3 == 0:
                    self.ax[j].axhline(c - 0.5, color="black", lw=2)
        self.gene_no = 0
        self.ax[0].set_title("Initial Bled Code", size=10)
        self.ax[1].set_title("Bayes Parallel Biased Bled Code", size=10)
        self.ax[0].set_yticks(ticks=np.arange(self.im_data[0].shape[0]), labels=nb.basic_info.use_channels)
        self.ax[1].set_xticks(ticks=np.arange(self.im_data[0].shape[1]))
        self.ax[1].set_xlabel("Round")
        self.fig.supylabel("Colour Channel")
        self.main_title = plt.suptitle("", x=(subplot_adjust[0] + subplot_adjust[1]) / 2)
        self.update_title()
        self.fig.canvas.mpl_connect("scroll_event", self.change_gene)
        self.change_norm()
        self.change_gene()  # plot rectangles
        plt.show()

    def change_gene(self, event=None):
        if event is not None:
            if event.button == "up":
                self.gene_no = (self.gene_no + 1) % self.n_genes
            else:
                self.gene_no = (self.gene_no - 1) % self.n_genes
        self.im_data = [val[:, :, self.gene_no] for val in self.im_data_3d]
        for i in range(self.n_images):
            # change image to different normalisation and change clim
            self.im[i].set_data(self.im_data[i] * self.colour_norm[i] if self.method == "raw" else self.im_data[i])
        self.update_title()
        self.im[-1].axes.figure.canvas.draw()

    def update_title(self):
        self.main_title.set_text(f"Gene {self.gene_no}, {self.gene_names[self.gene_no]} Bled Code")
        self.ax[1].set_xticklabels(self.use_rounds)
