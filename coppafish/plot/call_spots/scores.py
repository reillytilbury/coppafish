from matplotlib.widgets import TextBox, CheckButtons
from ...setup import Notebook
from ...call_spots import dot_product_score
from typing import Optional, List
from matplotlib.widgets import Button

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("dark_background")


class HistogramScore:
    ylim_tol = 0.2  # If fractional change in y limit is less than this, then leave it the same

    def __init__(
        self,
        nb: Notebook,
        hist_spacing: float = 0.01,
        show_plot: bool = True,
    ):
        """
        If method is anchor, this will show the histogram of `nb.call_spots.scores` with the option to
        view the histogram of the score computed using various other configurations of `background` fitting
        and `gene_efficiency`. This allows one to see how the these affect the score.

        There will also be the option to view the histograms shown for the anchor method.
        I.e. we compute the dot product score for the omp spots.

        Args:
            nb: *Notebook* containing at least `call_spots` page.
            hist_spacing: Initial width of bin in histogram.
            show_plot: Whether to run `plt.show()` or not.
        """
        # Add data
        self.gene_names = nb.call_spots.gene_names
        self.n_genes = self.gene_names.size
        self.n_rounds, self.n_channels = nb.call_spots.bled_codes.shape[1:]
        # Use all genes by default
        self.genes_use = np.arange(self.n_genes)

        # Get spot colors
        spot_colours_raw = nb.ref_spots.colours
        colour_norm_factor = nb.call_spots.colour_norm_factor
        spot_tile = nb.ref_spots.tile
        spot_colours = spot_colours_raw * colour_norm_factor[spot_tile]
        spot_colours_bg = np.repeat(np.percentile(spot_colours, 25, axis=1)[:, None, :], self.n_rounds, axis=1)
        spot_colours_bg_removed = spot_colours - spot_colours_bg

        # Bled codes saved to Notebook should already have L2 norm = 1 over used_channels and rounds
        bled_codes_initial = nb.call_spots.bleed_matrix[nb.call_spots.gene_codes]
        bled_codes_initial /= np.linalg.norm(bled_codes_initial, axis=(1, 2), keepdims=True)
        bled_codes = nb.call_spots.bled_codes

        # reshape to 2D
        spot_colours = spot_colours.reshape((spot_colours.shape[0], -1))
        spot_colours_bg_removed = spot_colours_bg_removed.reshape((spot_colours_bg_removed.shape[0], -1))
        bled_codes_initial = bled_codes_initial.reshape((bled_codes_initial.shape[0], -1))
        bled_codes = bled_codes.reshape((bled_codes.shape[0], -1))

        # Save score_dp for original score, without background removal, without gene efficiency, and without both
        self.n_plots = 4
        self.gene_no = nb.call_spots.dot_product_gene_no
        self.score = np.zeros((self.gene_no.size, self.n_plots), dtype=np.float32)
        self.method = "Anchor"

        self.use = np.isin(self.gene_no, self.genes_use)  # which spots to plot

        # DP score
        self.score[:, 0] = dot_product_score(spot_colours_bg_removed, bled_codes)[1]
        # DP score no background
        self.score[:, 1] = dot_product_score(spot_colours, bled_codes)[1]
        # DP score no gene efficiency
        self.score[:, 2] = dot_product_score(spot_colours_bg_removed, bled_codes_initial)[1]
        # DP score no background or gene efficiency
        self.score[:, 3] = dot_product_score(spot_colours, bled_codes_initial)[1]

        # Initialise plot
        self.fig, self.ax = plt.subplots(1, 1, figsize=(11, 5))
        self.subplot_adjust = [0.07, 0.85, 0.1, 0.93]
        self.fig.subplots_adjust(
            left=self.subplot_adjust[0],
            right=self.subplot_adjust[1],
            bottom=self.subplot_adjust[2],
            top=self.subplot_adjust[3],
        )
        self.ax.set_ylabel(r"Number of Spots")
        self.ax.set_xlabel(r"Score, $\Delta_s$")
        self.ax.set_title(f"Distribution of Scores for all {self.method} spots")

        # Plot histograms
        self.hist_spacing = hist_spacing
        hist_bins = np.arange(0, 1, self.hist_spacing)
        self.plots = [None] * self.n_plots
        default_colors = plt.rcParams["axes.prop_cycle"]._left
        for i in range(self.n_plots):
            y, x = np.histogram(self.score[self.use, i], hist_bins)
            x = x[:-1] + self.hist_spacing / 2  # so same length as x
            (self.plots[i],) = self.ax.plot(x, y, color=default_colors[i]["color"])
            if i > 0:
                self.plots[i].set_visible(False)

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, None)

        # Add text box to change score multiplier
        text_box_labels = ["Gene", "Histogram\nSpacing"]
        text_box_values = ["all", self.hist_spacing, ]
        text_box_funcs = [self.update_genes, self.update_hist_spacing]
        text_box_labels = text_box_labels[:2]
        text_box_values = text_box_values[:2]
        text_box_funcs = text_box_funcs[:2]
        self.text_boxes = [None] * len(text_box_labels)
        for i in range(len(text_box_labels)):
            text_ax = self.fig.add_axes(
                [
                    self.subplot_adjust[1] + 0.05,
                    self.subplot_adjust[2] + 0.15 * (len(text_box_labels) - i - 1),
                    0.05,
                    0.04,
                ]
            )
            self.text_boxes[i] = TextBox(
                text_ax, text_box_labels[i], text_box_values[i], color="k", hovercolor=[0.2, 0.2, 0.2]
            )
            self.text_boxes[i].cursor.set_color("r")
            label = text_ax.get_children()[0]  # label is a child of the TextBox axis
            if i == 0:
                label.set_position([0.5, 1.77])  # [x,y] - change here to set the position
            else:
                label.set_position([0.5, 2.75])
                # centering the text
            label.set_verticalalignment("top")
            label.set_horizontalalignment("center")
            self.text_boxes[i].on_submit(text_box_funcs[i])

        # Add buttons to add/remove score_dp histograms
        self.buttons_ax = self.fig.add_axes([self.subplot_adjust[1] + 0.02, self.subplot_adjust[3] - 0.45, 0.15, 0.5])
        plt.axis("off")
        self.button_labels = ["Dot Product "
                              "\nScore",
                              "No BG Removal",
                              "No Free Bled "
                              "\nCodes",
                              "No BG Removal"
                              "\nNo Free Bled "
                              "\nCodes",]
        label_checked = [True, False, False, False]
        self.buttons = CheckButtons(
            self.buttons_ax,
            self.button_labels,
            label_checked,
            label_props={
                "fontsize": [8] * self.n_plots,
                "color": [default_colors[i]["color"] for i in range(self.n_plots)],
            },
            frame_props={"edgecolor": ["w"] * self.n_plots},
            check_props={"facecolor": ["w"] * self.n_plots},
        )
        self.buttons.on_clicked(self.choose_plots)
        if show_plot:
            plt.show()

    def update(self, inds_update: Optional[List[int]] = None):
        ylim_old = self.ax.get_ylim()[1]  # To check whether we need to change y limit
        hist_bins = np.arange(0, 1, self.hist_spacing)
        if inds_update is None:
            inds_update = np.arange(len(self.plots))  # By default update all plots
        ylim_new = 0
        for i in np.arange(len(self.plots)):
            if i in inds_update:
                y, x = np.histogram(self.score[self.use, i], hist_bins)
                x = x[:-1] + self.hist_spacing / 2  # so same length as x
                self.plots[i].set_data(x, y)
            ylim_new = np.max([ylim_new, self.plots[i].get_ydata().max()])
        # self.ax.set_xlim(0, 1)
        if np.abs(ylim_new - ylim_old) / np.max([ylim_new, ylim_old]) < self.ylim_tol:
            ylim_new = ylim_old
        self.ax.set_ylim(0, ylim_new)
        if isinstance(self.genes_use, int):
            gene_label = f" matched to {self.gene_names[self.genes_use]}"
        else:
            gene_label = ""
        self.ax.set_title(f"Distribution of Scores for all {self.method} spots" + gene_label)
        self.ax.figure.canvas.draw()

    def update_genes(self, text):
        # TODO: If give 2+45 then show distribution for gene 2 and 45 on the same plot
        # Can select to view histogram of one gene or all genes
        if text.lower() == "all":
            g = "all"
        else:
            try:
                g = int(text)
                if g >= self.n_genes or g < 0:
                    warnings.warn(f"\nGene index needs to be between 0 and {self.n_genes}")
                    g = self.genes_use
            except (ValueError, TypeError):
                # if a string, check if is name of gene
                gene_names = list(map(str.lower, self.gene_names))
                try:
                    g = gene_names.index(text.lower())
                except ValueError:
                    # default to the best gene at this iteration
                    warnings.warn(f"\nGene given, {text}, is not valid")
                    if isinstance(self.genes_use, int):
                        g = self.genes_use
                    else:
                        g = "all"
        if g == "all":
            self.genes_use = np.arange(self.n_genes)
        else:
            self.genes_use = g
        self.use = np.isin(self.gene_no, self.genes_use)  # which spots to plot
        self.text_boxes[0].set_val(g)
        self.update()

    def update_hist_spacing(self, text):
        # Can select spacing of histogram
        try:
            hist_spacing = float(text)
        except (ValueError, TypeError):
            warnings.warn(f"\nScore multiplier given, {text}, is not valid")
            hist_spacing = self.hist_spacing
        if hist_spacing < 0:
            warnings.warn("Histogram spacing cannot be negative")
            hist_spacing = self.hist_spacing
        if hist_spacing < 0:
            warnings.warn("Histogram spacing cannot be negative")
            hist_spacing = self.hist_spacing
        if hist_spacing >= 1:
            warnings.warn("Histogram spacing cannot be >= 1")
            hist_spacing = self.hist_spacing
        self.hist_spacing = hist_spacing
        self.text_boxes[1].set_val(hist_spacing)
        self.update()

    def choose_plots(self, label):
        index = self.button_labels.index(label)
        self.plots[index].set_visible(not self.plots[index].get_visible())
        self.ax.figure.canvas.draw()


class ViewAllGeneHistograms:
    """
    Module to view all gene scores in a grid of n_genes histograms.
    """

    def __init__(self, nb):
        """
        Load in notebook and spots.
        Args:
            nb: Notebook object. Must have ref_spots page
        """
        self.nb = nb
        self.load_values()
        self.plot()

    def load_values(self, mode: str = "score"):
        """
        Load in values to be plotted.
        Args:
            mode: 'score', 'prob', 'prob_diff' or 'intensity'
        """
        self.mode = mode
        if mode == "score":
            values = self.nb.call_spots.dot_product_gene_score
        elif mode == "prob":
            values = np.max(self.nb.call_spots.gene_probabilities, axis=1)
        elif mode == "prob_diff":
            probs_sorted = np.sort(self.nb.call_spots.gene_probabilities, axis=1)
            values = probs_sorted[:, -1] - probs_sorted[:, -2]
        elif mode == "intensity":
            values = self.nb.call_spots.intensity
        else:
            raise ValueError("mode must be 'score', 'prob', 'prob_diff' or 'intensity'")

        gene_values = np.zeros((self.nb.call_spots.gene_names.shape[0], 0)).tolist()
        prob_assignments = np.argmax(self.nb.call_spots.gene_probabilities, axis=1)
        dot_product_assignments = self.nb.call_spots.dot_product_gene_no
        for g in range(len(gene_values)):
            if mode != "prob":
                gene_values[g] = values[dot_product_assignments == g]
            else:
                gene_values[g] = values[prob_assignments == g]

        self.gene_values = gene_values
        self.n_spots = np.array([len(gene_values[i]) for i in range(len(gene_values))])

    def plot(self):
        """
        Plot self.gene_values in a grid of histograms. Side length of grid is sqrt(n_genes).
        """
        # Delete any existing plots
        grid_dim = int(np.ceil(np.sqrt(len(self.nb.call_spots.gene_names))))
        if hasattr(self, "fig"):
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots(grid_dim, grid_dim, figsize=(10, 10))

        # Move subplots down to make room for the title and to the left to make room for the colourbar
        self.fig.subplots_adjust(top=0.9)
        self.fig.subplots_adjust(right=0.9)

        # Now loop through each subplot. If there are more subplots than genes, we want to delete the empty ones
        # We also want to plot the histogram of scores for each gene, and colour the histogram based on the number of
        # spots for that gene
        for i in range(grid_dim**2):
            # Choose the colour of the histogram based on the number of spots. We will use a log scale, with the
            # minimum number of spots being 1 and the maximum being 1000. Use a blue to red colourmap
            cmap = plt.get_cmap("coolwarm")
            norm = mpl.colors.Normalize(vmin=1, vmax=np.percentile(self.n_spots, 99))

            # Plot the histogram of scores for each gene
            r, c = i // grid_dim, i % grid_dim
            if i < len(self.nb.call_spots.gene_names):
                if self.n_spots[i] < 50:
                    n_bins = 10
                else:
                    n_bins = 50
                self.ax[r, c].hist(self.gene_values[i], bins=n_bins, color=cmap(norm(self.n_spots[i])))
                self.ax[r, c].set_title(self.nb.call_spots.gene_names[i])
                self.ax[r, c].set_xlim(0, 1)
                self.ax[r, c].set_xticks([])
                self.ax[r, c].set_yticks([])

            # Next we want to delete the empty plots
            else:
                self.ax[r, c].axis("off")
                self.ax[r, c].set_xticks([])
                self.ax[r, c].set_yticks([])

        # Add overall title, colourbar and buttons
        self.fig.suptitle("Distribution of Scores for Each Gene", fontsize=16)
        cax = self.fig.add_axes([0.95, 0.1, 0.03, 0.6])
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, label="Number of Spots")
        # Put label on the left of the colourbar
        cax.yaxis.set_label_position("left")
        self.add_buttons()
        self.fig.canvas.draw_idle()
        plt.show()

    def add_buttons(self):
        """
        Add buttons to the plot. Buttons are:
            - 'score': plot the distribution of scores
            - 'prob': plot the distribution of probabilities
            - 'prob_diff': plot the distribution of score differences
            - 'intensity': plot the distribution of intensities
        """
        # coords for colourbar are [0.95, 0.1, 0.03, 0.6], we want to put the buttons just above the colourbar

        # create axes for the buttons
        ax_intensity = self.fig.add_axes([0.95, 0.75, 0.03, 0.03])
        ax_prob_diff = self.fig.add_axes([0.95, 0.8, 0.03, 0.03])
        ax_prob = self.fig.add_axes([0.95, 0.85, 0.03, 0.03])
        ax_score = self.fig.add_axes([0.95, 0.9, 0.03, 0.03])

        # create the buttons, make them black. Hovering over will make them white
        self.button_intensity = Button(ax_intensity, "intensity", color="black", hovercolor="white")
        self.button_prob_diff = Button(ax_prob_diff, "prob_diff", color="black", hovercolor="white")
        self.button_prob = Button(ax_prob, "prob", color="black", hovercolor="white")
        self.button_score = Button(ax_score, "score", color="black", hovercolor="white")

        # connect the buttons to the update function. We need to ensure the event passed to update is a string
        self.button_intensity.on_clicked(lambda event: self.update("intensity"))
        self.button_prob_diff.on_clicked(lambda event: self.update("prob_diff"))
        self.button_prob.on_clicked(lambda event: self.update("prob"))
        self.button_score.on_clicked(lambda event: self.update("score"))

    def update(self, event):
        """
        Update the plot when a button is clicked.
        """
        self.mode = event
        self.load_values(mode=self.mode)
        self.plot()