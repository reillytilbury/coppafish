import numpy as np
from matplotlib import pyplot as plt
from ...setup import Notebook


class ViewFreeAndConstrainedBledCodes:
    def __init__(self, nb: Notebook, r: int = None, c: int = None, show: bool = True):
        """
        Function to plot the free and constrained bled codes for each spot for a given round and channel.

        This will show a grid of images of the free and constrained bled codes which have the target dye in the given
        round and channel.

        Scroll up and down to navigate through the rounds and channels.

        Args:
            nb: Notebook
                The notebook object. Should contain call_spots page.
            r: Round
                The relevant round.
            c: Channel
                The relevant channel.
            show: Bool (default=True)
                Whether to show the plot. If False, the plot is not shown. False only for testing purposes.
        """
        if r is None:
            r = 0
        if c is None:
            c = 0

        # get the data
        n_genes, n_rounds, n_channels_use = (len(nb.call_spots.gene_names), len(nb.basic_info.use_rounds),
                                             len(nb.basic_info.use_channels))

        config = nb.call_spots.associated_configs["call_spots"]
        if config["target_values"] is None:
            if n_channels_use == 7:
                config["target_values"] = [1, 1, 0.9, 0.7, 0.8, 1, 1]
            elif n_channels_use == 9:
                config["target_values"] = [1, 0.8, 0.2, 0.9, 0.6, 0.8, 0.3, 0.7, 1]
            else:
                raise ValueError("The target values should be provided in the config.")
        if config["d_max"] is None:
            if n_channels_use == 7:
                config["d_max"] = [0, 1, 3, 2, 4, 5, 6]
            elif n_channels_use == 9:
                config["d_max"] = [0, 1, 1, 3, 2, 4, 5, 5, 6]
            else:
                raise ValueError("The d_max values should be provided in the config.")
        target_values, d_max = config["target_values"], config["d_max"]
        gene_codes = nb.call_spots.gene_codes
        rc_scale = nb.call_spots.rc_scale.T
        free_bled_codes = nb.call_spots.free_bled_codes_tile_independent.transpose(0, 2, 1)
        free_bled_codes /= np.linalg.norm(free_bled_codes, axis=(1, 2))[:, None, None]
        constrained_bled_codes = nb.call_spots.bled_codes.transpose(0, 2, 1)
        code_image = np.zeros((n_genes, n_channels_use, 2 * n_rounds + 1)) * np.nan
        n_spots = np.zeros(n_genes, dtype=int)

        # populate code image and n_spots
        for g in range(n_genes):
            code_image[g, :, :n_rounds] = free_bled_codes[g]
            code_image[g, :, -n_rounds:] = constrained_bled_codes[g]
            n_spots[g] = np.sum(np.argmax(nb.call_spots.gene_probabilities, axis=1) == g)

        # add the attributes
        self.code_image = code_image
        self.r, self.c = r, c
        self.n_genes, self.n_rounds, self.n_channels_use = n_genes, n_rounds, n_channels_use
        self.use_channels = nb.basic_info.use_channels
        self.d_max, self.target_values = d_max, target_values
        self.rc_scale = rc_scale
        self.gene_codes = gene_codes
        self.gene_names = nb.call_spots.gene_names
        self.n_spots = n_spots

        # set up the plot
        self.n_row_cols = 4
        self.fig, self.ax = plt.subplots(self.n_row_cols, self.n_row_cols)
        for i, j in np.ndindex(self.n_row_cols, self.n_row_cols):
            self.ax[i, j].imshow(np.zeros((n_genes, 2 * n_rounds + 1)) * np.nan, cmap="viridis")
            self.ax[i, j].axis("off")

        # add cbar ax
        self.cbar_ax = self.fig.add_axes([0.92, 0.15, 0.02, 0.7])

        # connect the scroll event
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        # update the plot
        self.update_plot()

    def update_plot(self):
        """
        Function to plot the free and constrained bled codes for each spot for a given round and channel.
        """
        relevant_genes = np.where(self.gene_codes[:, self.r] == self.d_max[self.c])[0]
        n_spots_rc = self.n_spots[relevant_genes]

        # sort the genes by number of spots, and only keep the top n_row_cols ** 2 - 1
        relevant_genes = relevant_genes[np.argsort(n_spots_rc)[::-1][:self.n_row_cols ** 2 - 1]]
        fig, ax = self.fig, self.ax

        # clear the axes
        for i, j in np.ndindex(self.n_row_cols, self.n_row_cols):
            ax[i, j].clear()
            ax[i, j].axis("off")

        # plot the data
        for i, g in enumerate(relevant_genes):
            row, col = i // self.n_row_cols, i % self.n_row_cols
            ax[row, col].imshow(self.code_image[g], cmap="viridis", vmin=0, vmax=np.nanmax(self.code_image))
            ax[row, col].set_title(f"{self.gene_names[g]}, n={self.n_spots[g]}", fontsize=8)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            # add a white box around round r channel c
            ax[row, col].add_patch(plt.Rectangle((self.r - 0.5, self.c - 0.5), 1, 1,
                                                 edgecolor="white", facecolor="none"))
            # add a white box around round r + n_rounds + 1, channel c
            ax[row, col].add_patch(plt.Rectangle((self.r + self.n_rounds + 0.5, self.c - 0.5), 1, 1,
                                                 edgecolor="white", facecolor="none"))

        # add the round/channel scale
        rc_scale_centred = (self.rc_scale - np.mean(self.rc_scale)) / np.std(self.rc_scale)
        ax[-1, -1].imshow(rc_scale_centred, cmap="viridis", vmin=np.min(rc_scale_centred),
                          vmax=np.max(rc_scale_centred))
        ax[-1, -1].set_title("round/channel scale (centred)")
        ax[-1, -1].set_xlabel("Round")
        ax[-1, -1].set_ylabel("Channel")
        ax[-1, -1].set_xticks([])
        ax[-1, -1].set_yticks([])
        ax[-1, -1].add_patch(plt.Rectangle((self.r - 0.5, self.c - 0.5), 1, 1,
                                             edgecolor="white", facecolor="none"))

        # clear the colorbar
        self.cbar_ax.clear()
        plt.colorbar(ax[0, 0].imshow(self.code_image[relevant_genes[0]], cmap="viridis", vmin=0,
                                     vmax=np.nanmax(self.code_image)), cax=self.cbar_ax)

        # add the title
        plt.suptitle(f"Free and constrained bled codes for round {self.r}, channel {self.use_channels[self.c]}. \n"
                     f" Boost = {rc_scale_centred[self.c, self.r]:.2f}")
        fig.canvas.draw()

    def on_scroll(self, event):
        """
        Function to navigate through the rounds and channels.
        """
        increment = 1 if event.button == "up" else -1
        round_channel_list = [(r, c) for r in range(self.n_rounds) for c in range(self.n_channels_use)]
        index = round_channel_list.index((self.r, self.c))
        index_new = (index + increment) % len(round_channel_list)
        self.r, self.c = round_channel_list[index_new]
        self.update_plot()


class ViewTargetRegression:
    def __init__(self, nb: Notebook, r: int = None, c: int = None):
        if r is None:
            r = 0
        if c is None:
            c = 0

        # get the data
        n_genes, n_rounds, n_channels_use = nb.call_spots.bled_codes.shape
        config = nb.call_spots.associated_configs["call_spots"]
        if config["target_values"] is None:
            if n_channels_use == 7:
                config["target_values"] = [1, 1, 0.9, 0.7, 0.8, 1, 1]
            elif n_channels_use == 9:
                config["target_values"] = [1, 0.8, 0.2, 0.9, 0.6, 0.8, 0.3, 0.7, 1]
            else:
                raise ValueError("The target values should be provided in the config.")
        if config["d_max"] is None:
            if n_channels_use == 7:
                config["d_max"] = [0, 1, 3, 2, 4, 5, 6]
            elif n_channels_use == 9:
                config["d_max"] = [0, 1, 1, 3, 2, 4, 5, 5, 6]
            else:
                raise ValueError("The d_max values should be provided in the config.")
        target_values, d_max = config["target_values"], config["d_max"]
        gene_codes = nb.call_spots.gene_codes
        # get the free and constrained bled codes
        rc_scale = nb.call_spots.rc_scale
        free_bled_codes = nb.call_spots.free_bled_codes_tile_independent
        constrained_bled_codes = free_bled_codes * rc_scale[None, :, :]

        # get the number of spots per gene
        n_spots = np.zeros(n_genes, dtype=int)
        for g in range(n_genes):
            n_spots[g] = np.sum(np.argmax(nb.call_spots.gene_probabilities, axis=1) == g)

        # add the attributes
        self.r, self.c = r, c
        self.n_genes, self.n_rounds, self.n_channels_use = n_genes, n_rounds, n_channels_use
        self.use_channels = nb.basic_info.use_channels
        self.d_max, self.target_values = d_max, target_values
        self.free_bled_codes, self.constrained_bled_codes = free_bled_codes, constrained_bled_codes
        self.rc_scale = rc_scale
        self.gene_codes, self.gene_names = gene_codes, nb.call_spots.gene_names
        self.n_spots = n_spots

        # set up the plot
        self.fig, self.ax = plt.subplots(1, 3)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.update_plot()

    def update_plot(self):
        """
        Function to plot the free and constrained bled codes for each spot for a given round and channel.
        """
        relevant_genes = np.where(self.gene_codes[:, self.r] == self.d_max[self.c])[0]
        n_relevant_genes = len(relevant_genes)
        fig, ax = self.fig, self.ax

        # clear the axes
        for a in ax:
            a.clear()

        # set up the data
        x_coords = [np.random.rand(n_relevant_genes)] * 2
        y_coords = [self.free_bled_codes[relevant_genes, self.r, self.c],
                    self.constrained_bled_codes[relevant_genes, self.r, self.c]]
        titles = ["Free bled codes", "Constrained bled codes"]
        colours = ["cyan", "red"]

        # plot the data
        for i, a in enumerate(ax[:2]):
            a.scatter(x_coords[i], y_coords[i], c=colours[i], s=np.sqrt(self.n_spots[relevant_genes]), alpha=0.5)
            a.set_title(titles[i])
            a.set_xticks([])
            a.set_xlabel("Random x values")
            a.set_ylabel("Bled code value")
            if i == 0:
                a.set_ylim(0, 1.5 * np.max(self.free_bled_codes))
                a.set_yticks(np.round([0, 1.5 * np.max(self.free_bled_codes)], 2))
            else:
                # add a horizontal line for the target value
                a.axhline(self.target_values[self.c], color="white", linestyle="--")
                a.set_ylim(0, 1.5 * np.max(self.constrained_bled_codes))
                a.set_yticks(np.round([0, self.target_values[self.c], 1.5 * np.max(self.constrained_bled_codes)], 2))

        # add the round/channel scale
        ax[-1].imshow(self.rc_scale.T, cmap="viridis")
        ax[-1].set_title("round/channel scale")
        ax[-1].set_ylabel("Channel")
        ax[-1].set_xlabel("Round")
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        ax[-1].add_patch(plt.Rectangle((self.r - 0.5, self.c - 0.5), 1, 1,
                                       edgecolor="white", facecolor="none"))

        plt.suptitle(f"Target regression for round {self.r}, channel {self.use_channels[self.c]}")
        fig.canvas.draw()

    def on_scroll(self, event):
        """
        Function to navigate through the rounds and channels.
        """
        increment = 1 if event.button == "up" else -1
        round_channel_list = [(r, c) for r in range(self.n_rounds) for c in range(self.n_channels_use)]
        index = round_channel_list.index((self.r, self.c))
        index_new = (index + increment) % len(round_channel_list)
        self.r, self.c = round_channel_list[index_new]
        self.update_plot()


def view_tile_bled_codes(
    nb: Notebook,
    gene: int = 0,
    show: bool = True,
) -> None:
    """
    Function to plot the free bled codes for each tile for a given gene.
    Args:
        nb: Notebook
            The notebook object. Should contain call_spots page.
        gene: int
            The gene to plot.
        show: bool (default=True)
            Whether to show the plot. If False, the plot is not shown. False only for testing purposes.
    """
    free_bled_codes = nb.call_spots.free_bled_codes
    free_bled_codes_tile_indep = nb.call_spots.free_bled_codes_tile_independent
    gene_names = nb.call_spots.gene_names
    use_tiles = nb.basic_info.use_tiles

    # plot the free bled codes for each tile
    n_columns = 4
    _, _, n_rounds, n_channels_use = free_bled_codes.shape
    n_tiles = len(use_tiles)
    n_rows = int(np.ceil(n_tiles / n_columns)) + 1

    fig, ax = plt.subplots(n_rows, n_columns)
    for i, t in enumerate(use_tiles):
        row, col = i // n_columns, i % n_columns
        ax[row, col].imshow(free_bled_codes[gene, t].T, cmap="viridis")
        ax[row, col].set_title(f"Tile {t}")
        ax[row, col].set_xlabel("Round")
        ax[row, col].set_ylabel("Channel")
        ax[row, col].set_xticks([])
        ax[row, col].set_yticks([])
    for i in range(n_tiles, n_rows * n_columns - 1):
        row, col = i // n_columns, i % n_columns
        ax[row, col].axis("off")

    # add the tile independent bled code
    ax[-1, -1].imshow(free_bled_codes_tile_indep[gene].T, cmap="viridis")
    ax[-1, -1].set_title("Tile independent")
    ax[-1, -1].set_xlabel("Round")
    ax[-1, -1].set_ylabel("Channel")
    ax[-1, -1].set_xticks([])
    ax[-1, -1].set_yticks([])

    plt.suptitle(f"Free bled codes for gene {gene_names[gene]}")
    if show:
        plt.show()


def ViewTileScaleRegression(
    nb: Notebook,
    t: int = None,
    show: bool = True,
) -> None:
    """
    Plotter to show the regression of the tile scale factor for tile t for all rounds and channels.

    Args:
        nb: Notebook
            The notebook object. Should contain call_spots page.
        t: int (default=0)
            The tile to plot.
        show: bool (default=True)
            Whether to show the plot. If False, the plot is not shown. False only for testing purposes.

    """
    if t is None:
        t = nb.basic_info.use_tiles[0]
    tile_scale = nb.call_spots.tile_scale
    gene_codes = nb.call_spots.gene_codes
    config = nb.call_spots.associated_configs["call_spots"]
    n_channels_use = len(nb.basic_info.use_channels)
    if config["target_values"] is None:
        if n_channels_use == 7:
            config["target_values"] = [1, 1, 0.9, 0.7, 0.8, 1, 1]
        elif n_channels_use == 9:
            config["target_values"] = [1, 0.8, 0.2, 0.9, 0.6, 0.8, 0.3, 0.7, 1]
        else:
            raise ValueError("The target values should be provided in the config.")
    if config["d_max"] is None:
        if n_channels_use == 7:
            config["d_max"] = [0, 1, 3, 2, 4, 5, 6]
        elif n_channels_use == 9:
            config["d_max"] = [0, 1, 1, 3, 2, 4, 5, 5, 6]
        else:
            raise ValueError("The d_max values should be provided in the config.")
    d_max = config["d_max"]
    target_bled_codes = nb.call_spots.bled_codes
    free_bled_codes = nb.call_spots.free_bled_codes
    gene_no = np.argmax(nb.call_spots.gene_probabilities, axis=1)
    n_spots = np.array([np.sum(gene_no == i) for i in range(len(gene_no))])
    use_channels = nb.basic_info.use_channels
    n_tiles, n_rounds, n_channels_use = tile_scale.shape

    # plot the tile scale factors for tile
    fig, ax = plt.subplots(n_rounds, n_channels_use)
    for r, c in np.ndindex(n_rounds, n_channels_use):
        relevant_genes = np.where(gene_codes[:, r] == d_max[c])[0]
        n_spots_rc = n_spots[relevant_genes]
        x = free_bled_codes[relevant_genes, t, r, c]
        y = target_bled_codes[relevant_genes, r, c]
        sizes = np.sqrt(n_spots_rc)
        ax[r, c].scatter(x, y, s=sizes)
        ax[r, c].plot(x, tile_scale[t, r, c] * x, color="red")
        ax[r, c].set_title(f"r {r}, c {use_channels[c]}", fontsize=8)
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].set_xlim(0, free_bled_codes[:, t].max())
        ax[r, c].set_ylim(0, target_bled_codes.max())
        if r == n_rounds - 1:
            ax[r, c].set_xlabel("$D_{gtrc}$")
        if c == 0:
            ax[r, c].set_ylabel("$K_{grc}$")
    plt.suptitle(
        "Each tile scale $Q_{trc}$ is chosen so that the loss function "
        "$L(Q_{trc}) = \sum_{g \in G_{rc}} \sqrt{N_g}(D_{gtrc}Q_{trc} - K_{grc})^2$ is minimised, \n"
        "where $G_{rc}$ is the set of genes with gene code $d_{max}(c)$ in round r, $K_{grc} = E_{grc}V_{rc}$ "
        "is the constrained bled code for gene g, $D_g$ is the tile-dependent bled code for gene g, "
        "and $V_{rc}$ is the round/channel scale factor."
    )
    if show:
        plt.show()


def ViewScaleFactors(
    nb: Notebook,
    show: bool = True,
) -> None:
    """
    Function to plot the tile scale factors for each tile, round and channel.
    Args:
        nb: Notebook
            The notebook object. Should contain call_spots page.
        show: bool (default=True)
            Whether to show the plot. If False, the plot is not shown. False only for testing purposes.
    """
    tile_scale = nb.call_spots.tile_scale
    rc_scale = nb.call_spots.rc_scale
    use_tiles, use_rounds, use_channels = nb.basic_info.use_tiles, nb.basic_info.use_rounds, nb.basic_info.use_channels
    tile_scale = tile_scale[use_tiles]
    relative_scale = tile_scale / rc_scale[None, :, :]
    n_tiles, n_rounds, n_channels_use = tile_scale.shape
    tile_scale = tile_scale.reshape((n_tiles * n_rounds, n_channels_use))
    relative_scale = relative_scale.reshape((n_tiles * n_rounds, n_channels_use))

    # plot the scale factors
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(rc_scale, cmap="viridis")
    ax[0].set_xticks(np.arange(n_channels_use), labels=use_channels)
    ax[0].set_yticks(np.arange(n_rounds), labels=use_rounds)
    ax[0].set_title("round/channel factors $V_{rc}$")
    ax[0].set_xlabel("Channel")
    ax[0].set_ylabel("Round")
    plt.colorbar(ax[0].imshow(rc_scale, cmap="viridis"), ax=ax[0])
    for i, scale in enumerate([tile_scale, relative_scale]):
        i += 1
        ax[i].imshow(scale, cmap="viridis")
        ax[i].set_xticks(np.arange(n_channels_use), labels=use_channels)
        ax[i].set_yticks(
            np.arange(n_tiles * n_rounds), labels=[f"T{t}, R{r}" for t, r in np.ndindex(n_tiles, n_rounds)], fontsize=8
        )
        ax[i].set_title("tile scale factors $Q_{t,r,c}$" if i == 1 else "relative scale factors $Q_{t,r,c}/V_{rc}$")
        ax[i].set_xlabel("Channel")
        ax[i].set_ylabel("Tile x Round")
        # add horizontal red lines at each tile
        for t in range(1, n_tiles):
            ax[i].axhline(t * n_rounds - 0.5, color="red", linestyle="--")
        # add colorbar
        plt.colorbar(ax[i].imshow(scale, cmap="viridis"), ax=ax[i])
    if show:
        plt.show()

