import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources  # Python 3.10 support

from ...setup.notebook import Notebook


class Keywords:
    # All keywords must be lowercase to work
    QUIT = ("q", "quit")
    HELP = ("help", "h")
    REDRAW = ("r", "redraw")
    METHOD = ("v", "view")
    Z_UP = ("u", "up")
    Z_DOWN = ("d", "down")
    Z_MIN = ("zmin", "z_min")
    Z_MAX = ("zmax", "z_max")
    TOGGLE_LEGEND = ("l", "legend")


class Viewer2D:
    def tuple_to_str(self, array: Tuple[str]) -> str:
        output = ""
        for i, value in enumerate(list(array)):
            if i != 0:
                output += " or "
            output += f"'{str(value)}'"
        return output

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
            Keywords.QUIT: f"{self.tuple_to_str(Keywords.QUIT)} close the Viewer2D",
            Keywords.HELP: f"{self.tuple_to_str(Keywords.HELP)} show available commands",
            Keywords.HELP: f"{self.tuple_to_str(Keywords.HELP)} [command] get help for given command",
            Keywords.REDRAW: f"{self.tuple_to_str(Keywords.REDRAW)} manually redraw the Viewer",
            Keywords.METHOD: f"{self.tuple_to_str(Keywords.METHOD)} [method] show the given gene calling method. Can "
            + "be 'anchor' (default), 'probs', or 'omp'",
            Keywords.Z_UP: f"{self.tuple_to_str(Keywords.Z_UP)} move up by one z plane",
            Keywords.Z_DOWN: f"{self.tuple_to_str(Keywords.Z_DOWN)} move down by one z plane",
            Keywords.Z_MIN: f"{self.tuple_to_str(Keywords.Z_MIN)} [value] set the minimum viewed z to given value",
            Keywords.Z_MAX: f"{self.tuple_to_str(Keywords.Z_MAX)} [value] set the maximum viewed z to given value",
            Keywords.TOGGLE_LEGEND: f"{self.tuple_to_str(Keywords.TOGGLE_LEGEND)} toggle gene legend view",
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
        self.legend_symbols = gene_legend_info["mpl_symbol"]
        self.legend_gene_no = np.ones(n_legend_genes, dtype=int)
        # n_genes x 3 (R, G, and B values each between 0 and 1)
        self.gene_color = np.zeros((len(nb.call_spots.gene_names), 3))
        for i in range(n_legend_genes):
            self.legend_gene_no[i] = np.where(self.gene_names == gene_legend_info["GeneNames"][i])[0][0]
            self.gene_color[self.legend_gene_no[i]] = [
                gene_legend_info.loc[i, "ColorR"],
                gene_legend_info.loc[i, "ColorG"],
                gene_legend_info.loc[i, "ColorB"],
            ]

        # Default settings for the Viewer
        self.z_min: int = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
        self.z_max: int = self.z_min
        self.legend_show = False
        # These numbers do not matter in magnitude, but do in relative terms
        self.legend_gene_separation_horizontal = 4
        self.legend_gene_separation_vertical = 1

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
        plt.show()

    def _draw(self) -> None:
        plt.style.use("default")
        plt.ion()

    def _review(self) -> None:
        self._close()
        self._draw()
        self.view()

    def _interpret_command(self, command: str) -> None:
        """Handles user commands sent to the Viewer2D through the terminal."""
        if command == "":
            print(f"No command given.")
            return
        keyword = command.split()[0].lower()
        if not self._keyword_exists(keyword):
            print(f"Unknown command {keyword}")
            return
        args = command.split()[1:]
        for i, arg in enumerate(args):
            args[i] = arg.lower()
        if keyword in Keywords.QUIT:
            self._exit()
        elif keyword in Keywords.HELP:
            if len(args) == 0:
                print(f"Viewer2D help:")
                for _, description in self.commands.items():
                    print(description)
            elif self._keyword_exists(args[0]):
                # Show help for given keyword
                print(self._get_keyword_description(args[0]))
            else:
                print(f"Unknown command '{args[0]}'")
        elif keyword in Keywords.REDRAW:
            self._review()
        elif keyword in Keywords.Z_UP:
            self.z_min += 1
            self.z_max += 1
            self._review()
        elif keyword in Keywords.Z_DOWN:
            self.z_min -= 1
            self.z_max -= 1
            self._review()
        elif keyword in Keywords.Z_MIN:
            try:
                new_z_min = int(args[0])
                self.z_min = min([self.z_max, new_z_min])
                self._review()
            except ValueError:
                print(f"Cannot assign {args[0]} to z min")
                return
        elif keyword in Keywords.Z_MAX:
            try:
                new_z_max = int(args[0])
                self.z_max = max([self.z_min, new_z_max])
                self._review()
            except:
                print(f"Cannot assign {args[0]} to z max")
                return

    def _get_keyword_description(self, keyword: str) -> str:
        for keywords, description in self.commands.items():
            if keyword in keywords:
                return description

    def _keyword_exists(self, keyword: str) -> bool:
        for keywords, _ in self.commands.items():
            if keyword in keywords:
                return True
        return False

    def _exit(self) -> None:
        plt.style.use("default")
        print(f"Exiting Viewer2D")
        sys.exit()
