import math as maths
import sys
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ... import log
from ...setup.notebook import Notebook

try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources  # Python 3.10 support


class Viewer2D:
    class Keywords:
        # All keywords must be lowercase.
        Z_UP = ("u", "up")
        Z_DOWN = ("d", "down")
        Z_SHIFT = ("shift",)
        Z_MIN = ("min", "zmin")
        Z_MAX = ("max", "zmax")

        SCORE_MIN = ("scoremin", "score_min")
        SCORE_MAX = ("scoremax", "score_max")
        TOGGLE_LEGEND = ("l", "legend")
        TOGGLE_DARK_LIGHT_MODE = ("toggledark",)
        REDRAW = ("r", "redraw")

        METHOD = ("m", "method")
        SHOW_ALL = ("all",)
        SHOW_NONE = ("none",)
        TOGGLE_GENE = ("toggle",)
        TOGGLE_GENE_COLOUR = ("togglec", "togglecolour", "togglecolor")

        HELP = ("help", "h")
        COUNT_GENES = ("count",)

        QUIT = ("q", "quit", "exit")

        # Each keyword is placed in a subheading for a clearer 'help' view. Any unassigned keywords are placed in the
        # last subheading
        SECTIONS = {
            "Navigating": (Z_UP, Z_DOWN, Z_SHIFT, Z_MIN, Z_MAX),
            "Viewing": (SCORE_MIN, SCORE_MAX, TOGGLE_LEGEND, TOGGLE_DARK_LIGHT_MODE, REDRAW),
            "Gene selection": (METHOD, SHOW_ALL, SHOW_NONE, TOGGLE_GENE, TOGGLE_GENE_COLOUR),
            "Information": (HELP, COUNT_GENES),
            "Others": (QUIT,),
        }

    class Methods:
        anchor = "anchor"
        probability = "probs"
        omp = "omp"

    def _tuple_to_str(self, array: Tuple[str]) -> str:
        output = ""
        for i, value in enumerate(list(array)):
            if i != 0:
                output += " or "
            output += f"'{str(value)}'"
        return output

    def _str_to_method(self, command: str) -> Union[Methods, None]:
        if command == self.Methods.anchor:
            return self.Methods.anchor
        elif command == self.Methods.probability:
            return self.Methods.probability
        elif command == self.Methods.omp:
            return self.Methods.omp
        else:
            return None

    def _method_to_str(self, method: Methods) -> str:
        return str(method)

    def __init__(
        self, nb: Union[Notebook, str], gene_marker_file: Optional[str] = None, auto_view: bool = True
    ) -> None:
        """
        Create a new Viewer2D instance for the given notebook.

        Args:
            nb (Notebook or str): notebook or path to notebook.
            gene_marker_file (str, optional): specify how genes are plotted by giving a file path to a .csv marker
                file. Any genes not present in this file are not plotted. Default: one contained in the repository.
            auto_view (str, optional): whether to immediately open the Viewer2D once drawn. Default: true.
        """
        if isinstance(nb, str):
            nb = Notebook(nb)
        if not nb.has_page("call_spots"):
            raise ValueError(
                f"Given notebook does not have call spots. Complete up to call spots before opening the Viewer2D"
            )

        self.commands = {
            self.Keywords.QUIT: f"{self._tuple_to_str(self.Keywords.QUIT)} close the Viewer2D",
            self.Keywords.HELP: f"{self._tuple_to_str(self.Keywords.HELP)} show available commands",
            self.Keywords.HELP: f"{self._tuple_to_str(self.Keywords.HELP)} [command] get help for given command",
            self.Keywords.REDRAW: f"{self._tuple_to_str(self.Keywords.REDRAW)} manually redraw the Viewer",
            self.Keywords.METHOD: f"{self._tuple_to_str(self.Keywords.METHOD)} [method] show the given gene calling "
            + "method. Can be 'anchor', 'probs' for Von-Mises probabilities, or 'omp'",
            self.Keywords.SHOW_ALL: f"{self._tuple_to_str(self.Keywords.SHOW_ALL)} show all genes",
            self.Keywords.SHOW_NONE: f"{self._tuple_to_str(self.Keywords.SHOW_NONE)} show no genes",
            self.Keywords.TOGGLE_GENE: f"{self._tuple_to_str(self.Keywords.TOGGLE_GENE)} [gene] toggle the given "
            + "gene name on or off",
            self.Keywords.TOGGLE_GENE_COLOUR: f"{self._tuple_to_str(self.Keywords.TOGGLE_GENE_COLOUR)} [gene] toggle "
            + "the given gene name's colour on or off",
            self.Keywords.Z_UP: f"{self._tuple_to_str(self.Keywords.Z_UP)} move up by one z plane",
            self.Keywords.Z_DOWN: f"{self._tuple_to_str(self.Keywords.Z_DOWN)} move down by one z plane",
            self.Keywords.Z_SHIFT: f"{self._tuple_to_str(self.Keywords.Z_SHIFT)} [value] move up by value z plane(s). "
            + "Can be a negative value",
            self.Keywords.Z_MIN: f"{self._tuple_to_str(self.Keywords.Z_MIN)} [value] set the minimum viewed z to value",
            self.Keywords.Z_MAX: f"{self._tuple_to_str(self.Keywords.Z_MAX)} [value] set the maximum viewed z to value",
            self.Keywords.SCORE_MIN: f"{self._tuple_to_str( self.Keywords.SCORE_MIN)} [value] set the minimum score "
            + "to value",
            self.Keywords.SCORE_MAX: f"{self._tuple_to_str( self.Keywords.SCORE_MAX)} [value] set the maximum score "
            + "to value",
            self.Keywords.TOGGLE_LEGEND: f"{self._tuple_to_str(self.Keywords.TOGGLE_LEGEND)} toggle gene legend",
            self.Keywords.TOGGLE_DARK_LIGHT_MODE: f"{self._tuple_to_str(self.Keywords.TOGGLE_DARK_LIGHT_MODE)} toggle "
            + "between light and dark mode plot",
            self.Keywords.COUNT_GENES: f"{self._tuple_to_str(self.Keywords.COUNT_GENES)} count all visible genes",
        }

        if gene_marker_file is None:
            gene_marker_file = importlib_resources.files("coppafish.plot.results_viewer").joinpath("gene_color.csv")
        gene_legend_info = pd.read_csv(gene_marker_file)
        # Remove any genes from the legend which were not used in this notebook
        n_legend_genes = len(gene_legend_info["GeneNames"])
        unused_genes = []
        for i in range(n_legend_genes):
            if gene_legend_info["GeneNames"][i] not in nb.call_spots.gene_names:
                unused_genes.append(i)
        gene_legend_info = gene_legend_info.drop(unused_genes)
        # We want the data frame to be indexed from 0 to n_legend_genes-1
        gene_legend_info = gene_legend_info.reset_index(drop=True)

        n_legend_genes = len(gene_legend_info["GeneNames"])
        self.legend_symbols = gene_legend_info["mpl_symbol"].to_numpy().astype(str)
        self.legend_gene_no = np.ones(n_legend_genes, dtype=int)
        self.legend_gene_colours = np.zeros((n_legend_genes, 3), dtype=float)
        # n_genes x 3 (R, G, and B values each between 0 and 1)
        self.gene_colours = np.zeros((len(nb.call_spots.gene_names), 3))
        self.gene_names = nb.call_spots.gene_names
        for i in range(n_legend_genes):
            self.legend_gene_no[i] = np.where(self.gene_names == gene_legend_info["GeneNames"][i])[0][0]
            self.legend_gene_colours[i, 0] = float(gene_legend_info.loc[i, "ColorR"])
            self.legend_gene_colours[i, 1] = float(gene_legend_info.loc[i, "ColorG"])
            self.legend_gene_colours[i, 2] = float(gene_legend_info.loc[i, "ColorB"])
            self.gene_colours[self.legend_gene_no[i]] = [
                gene_legend_info.loc[i, "ColorR"],
                gene_legend_info.loc[i, "ColorG"],
                gene_legend_info.loc[i, "ColorB"],
            ]

        # Keep gene positions and scores inside the Viewer2D instance.
        self.anchor_global_yxz = nb.ref_spots.local_yxz + nb.stitch.tile_origin[nb.ref_spots.tile]
        self.anchor_gene_no = nb.ref_spots.gene_no
        self.anchor_score = nb.ref_spots.dot_product_gene_score
        self.probs_global_yxz = self.anchor_global_yxz
        self.probs_gene_no = nb.ref_spots.gene_probs.argmax(1)
        self.probs_score = nb.ref_spots.gene_probs.max(1)
        self.omp_available = nb.has_page("omp")
        if self.omp_available:
            self.omp_global_yxz = nb.omp.local_yxz + nb.stitch.tile_origin[nb.omp.tile]
            self.omp_gene_no = nb.omp.gene_no
            self.omp_score = nb.omp.scores
        tile_shape = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, len(nb.basic_info.use_z)])
        self.minimum_global_yxz = np.nanmin(nb.stitch.tile_origin, axis=0)
        self.maximum_global_yxz = np.nanmax(nb.stitch.tile_origin, axis=0) + tile_shape

        # Default settings for the Viewer.
        self.method_selected = self.Methods.omp if self.omp_available else self.Methods.anchor
        self.method_score_thresholds = {
            self.Methods.anchor: (0.75, 1.00),
            self.Methods.probability: (0.90, 1.00),
            self.Methods.omp: (0.30, 1.00),
        }
        self.z_min: int = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
        self.z_max: int = self.z_min + 1
        self.show_gene_no = np.full_like(self.gene_names, fill_value=True, dtype=bool)
        self.legend_show = self.legend_gene_no.size > 0
        self.legend_gene_separation_horizontal = 1
        self.legend_gene_separation_vertical = 1
        # If false, the plot is not shown to the user, even when updated.
        self.view_plot = True
        self.dark_mode = True

        self._draw()
        if auto_view:
            self.view()

        try:
            while True:
                inp = input(f"Viewer2D command: ")
                self._interpret_command(inp.strip())
                print("")
        except KeyboardInterrupt:
            self._exit()

    def _close(self) -> None:
        plt.close()

    def view(self) -> None:
        self._maximise_plot()
        plt.show()

    def _draw(self) -> None:
        plt.style.use("dark_background") if self.dark_mode else plt.style.use("default")
        gridspec_kw = {"width_ratios": [1, 7]} if self.legend_show else None
        fig, axes = plt.subplots(1, 2 if self.legend_show else 1, squeeze=False, gridspec_kw=gridspec_kw)
        if self.legend_show:
            ax_legend: plt.Axes = axes[0, 0]
            gene_count_height = maths.ceil(
                maths.sqrt(
                    9
                    * self.legend_gene_no.size
                    * self.legend_gene_separation_horizontal
                    / (2 * self.legend_gene_separation_vertical)
                )
            )
            gene_count_width = 0
            while (gene_count_width * gene_count_height) < self.legend_gene_no.size:
                gene_count_width += 1
            for i in range(self.legend_gene_no.size):
                x = self.legend_gene_separation_horizontal * (i % gene_count_width)
                y = self.legend_gene_separation_vertical * (i // gene_count_width)
                ax_legend.scatter(x, y, s=300, c=self.legend_gene_colours[[i]], marker=self.legend_symbols[i])
                ax_legend.annotate(self.gene_names[self.legend_gene_no[i]], (x, y), size="small")
                ax_legend.set_title(f"Gene legend")
            self._delete_splines_for(ax_legend)
            self._delete_ticks_for(ax_legend)
        ax_genes: plt.Axes = axes[-1, -1]
        if self.method_selected == self.Methods.anchor:
            global_yxz = self.anchor_global_yxz
            gene_no = self.anchor_gene_no
            score = self.anchor_score
        elif self.method_selected == self.Methods.probability:
            global_yxz = self.probs_global_yxz
            gene_no = self.probs_gene_no
            score = self.probs_score
        elif self.method_selected == self.Methods.omp:
            global_yxz = self.omp_global_yxz
            gene_no = self.omp_gene_no
            score = self.omp_score
        else:
            raise ValueError(f"Unkown selected method: {self.method_selected}")

        # Draw all genes for the selected method.
        in_z_range = np.logical_and(global_yxz[:, 2] >= self.z_min, global_yxz[:, 2] <= self.z_max)
        in_score_range = np.logical_and(
            score >= self.method_score_thresholds[self.method_selected][0],
            score <= self.method_score_thresholds[self.method_selected][1],
        )
        self.marker_count = 0
        # We must for loop over genes because each gene has a different marker.
        for g in np.unique(gene_no):
            if not self.show_gene_no[g]:
                continue
            is_gene_g = gene_no == g
            if np.isin(g, self.legend_gene_no):
                marker = self.legend_symbols[self.legend_gene_no == g].item()
            else:
                log.warn(f"Gene number {g} does not have an assigned marker")
                marker = ""
            keep = in_z_range * is_gene_g * in_score_range
            self.marker_count += keep.sum()
            ax_genes.scatter(
                global_yxz[keep, 1],
                global_yxz[keep, 0],
                s=50,
                c=self.gene_colours[[g]],
                marker=marker,
            )
        score_str = self.method_score_thresholds[self.method_selected]
        score_str = str(score_str[0]) + ", " + str(score_str[1])
        method_str = self._method_to_str(self.method_selected)
        ax_genes.set_xlim(self.minimum_global_yxz[1], self.maximum_global_yxz[1])
        ax_genes.set_ylim(self.minimum_global_yxz[0], self.maximum_global_yxz[0])
        ax_genes.set_title(f"Gene calls, {method_str}, z=[{self.z_min}, {self.z_max}], scores=[{score_str}]")
        fig.set_layout_engine("constrained")
        plt.ion()

    def _print_helper(self) -> None:
        for subheading in self.Keywords.SECTIONS:
            print(subheading.center(40, "="))
            for keywords in self.Keywords.SECTIONS[subheading]:
                print(self.commands[keywords])

    def _interpret_command(self, command: str) -> None:
        """Handles user commands sent to the Viewer2D through the terminal."""
        if command == "":
            print(f"No command given.")
            return
        keyword = command.split()[0].lower()
        if not self._keyword_exists(keyword):
            self._unknown_command_error(keyword)
            return
        args = command.split()[1:]
        for i, arg in enumerate(args):
            args[i] = arg.lower()
        if keyword in self.Keywords.QUIT:
            self._exit()
        elif keyword in self.Keywords.HELP:
            if len(args) == 0:
                self._print_helper()
            elif self._keyword_exists(args[0]):
                # Show help for given keyword
                print(self._get_keyword_description(args[0]))
            else:
                self._unknown_command_error(args[0])
            return
        elif keyword in self.Keywords.REDRAW:
            pass
        elif keyword in self.Keywords.Z_UP:
            self.z_min += 1
            self.z_max += 1
        elif keyword in self.Keywords.Z_DOWN:
            self.z_min -= 1
            self.z_max -= 1
        elif keyword in self.Keywords.Z_SHIFT:
            if len(args) == 0:
                self._argument_not_given("z shift amount")
                return
            else:
                try:
                    shift = int(args[0])
                    self.z_min += shift
                    self.z_max += shift
                    self._cap_z_planes()
                except ValueError:
                    self._assignment_error(args[0], "z shift")
                    return
        elif keyword in self.Keywords.Z_MIN:
            try:
                new_z_min = int(args[0])
                self.z_min = min([self.z_max, new_z_min])
                self._cap_z_planes()
            except ValueError:
                self._assignment_error(args[0], "z min")
                return
        elif keyword in self.Keywords.Z_MAX:
            try:
                new_z_max = int(args[0])
                self.z_max = max([self.z_min, new_z_max])
                self._cap_z_planes()
            except ValueError:
                self._assignment_error(args[0], "z max")
                return
        elif keyword in self.Keywords.TOGGLE_LEGEND:
            self.legend_show = not self.legend_show
        elif keyword in self.Keywords.METHOD:
            if len(args) == 0:
                self._argument_not_given("method")
                return
            method = self._str_to_method(args[0])
            if method is None:
                self._argument_invalid("method", args[0])
                return
            elif method == self.Methods.omp and not self.omp_available:
                print(f"omp method not available")
                return
            self.method_selected = method
        elif keyword in self.Keywords.SCORE_MIN:
            if len(args) == 0:
                self._argument_not_given("new score min")
                return
            try:
                new_score_min = float(args[0])
            except ValueError:
                self._assignment_error(args[0], "score min")
                return
            old_score_max = self.method_score_thresholds[self.method_selected][1]
            self.method_score_thresholds[self.method_selected] = (new_score_min, old_score_max)
        elif keyword in self.Keywords.SCORE_MAX:
            if len(args) == 0:
                self._argument_not_given("new score max")
                return
            try:
                new_score_max = float(args[0])
            except ValueError:
                self._assignment_error(args[0], "score max")
                return
            old_score_min = self.method_score_thresholds[self.method_selected][0]
            self.method_score_thresholds[self.method_selected] = (old_score_min, new_score_max)
        elif keyword in self.Keywords.TOGGLE_DARK_LIGHT_MODE:
            self.dark_mode = not self.dark_mode
        elif keyword in self.Keywords.COUNT_GENES:
            print(f"Total gene reads shown: {self.marker_count}")
            return
        elif keyword in self.Keywords.SHOW_ALL:
            self.show_gene_no[:] = True
        elif keyword in self.Keywords.SHOW_NONE:
            self.show_gene_no[:] = False
        elif keyword in self.Keywords.TOGGLE_GENE:
            if len(args) == 0:
                self._argument_not_given("gene name")
                return
            gene_no = self._get_unique_gene_starting_with(args[0])
            if gene_no is None:
                print(f"Could not find unique gene called {args[0]}")
                return
            self.show_gene_no[gene_no] = not self.show_gene_no[gene_no]
        elif keyword in self.Keywords.TOGGLE_GENE_COLOUR:
            if len(args) == 0:
                self._argument_not_given("gene name")
                return
            gene_no = self._get_unique_gene_starting_with(args[0])
            if gene_no is None:
                print(f"Could not find unique gene called {args[0]}")
                return
            self._toggle_gene_colour(gene_no)
        else:
            raise LookupError(f"Should not reach here")
        self._close()
        self._draw()
        if self.view_plot:
            self.view()

    def _cap_z_planes(self) -> None:
        self.z_min = max([self.minimum_global_yxz[2], self.z_min])
        self.z_max = min([self.maximum_global_yxz[2], self.z_max])

    def _unknown_command_error(self, given_commmand: str) -> None:
        print(f"Unknown command '{given_commmand}'")

    def _assignment_error(self, given_input: str, to_assign: str) -> None:
        print(f"Cannot assign {to_assign} to '{given_input}' given")

    def _argument_invalid(self, argument_name: str, given_value: str) -> None:
        print(f"'{given_value}' is invalid for {argument_name}")

    def _argument_not_given(self, argument_name: str) -> None:
        print(f"Argument {argument_name} not given")

    def _get_keyword_description(self, keyword: str) -> str:
        for keywords, description in self.commands.items():
            if keyword in keywords:
                return description

    def _keyword_exists(self, keyword: str) -> bool:
        for keywords, _ in self.commands.items():
            if keyword in keywords:
                return True
        return False

    def _toggle_gene_colour(self, gene_no: int) -> None:
        """Toggle all genes with the same colour as the given gene number."""
        colour = self.gene_colours[gene_no]
        gene_numbers = []
        for i, c in enumerate(self.gene_colours):
            if np.allclose(c, colour):
                gene_numbers.append(i)
        if self.show_gene_no[gene_numbers].any() and (~self.show_gene_no[gene_numbers]).any():
            self.show_gene_no[gene_numbers] = False
            return
        self.show_gene_no[gene_numbers] = not self.show_gene_no[gene_numbers[0]]

    def _get_unique_gene_starting_with(self, value: str) -> Union[int, None]:
        """Returns None if too many gene names start with value or none do."""
        gene_numbers = self._get_genes_starting_with(value)
        if gene_numbers.size > 1 or gene_numbers.size == 0:
            return None
        return gene_numbers[0].item()

    def _get_genes_starting_with(self, value: str) -> np.ndarray[int]:
        starts_with = np.array([gene_name.lower().startswith(value) for gene_name in self.gene_names])
        return np.where(starts_with)[0]

    def _maximise_plot(self) -> None:
        # This works on Windows. If it does not work on Linux... ¯\_(ツ)_/¯
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

    def _delete_splines_for(self, ax: plt.Axes) -> None:
        ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

    def _delete_ticks_for(self, ax: plt.Axes) -> None:
        """Remove x and y ticks for given axes"""
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    def _exit(self) -> None:
        plt.style.use("default")
        print(f"Exiting Viewer2D")
        sys.exit()
