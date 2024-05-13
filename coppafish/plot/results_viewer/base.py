import os
import pandas as pd
import numpy as np
import napari
import time
import skimage
import warnings
import matplotlib.pyplot as plt
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider, QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow, QSlider
from napari.layers.points import Points
from napari.layers.points._points_constants import Mode

try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources  # Python 3.10 support
from typing import Optional

from . import legend
from .hotkeys import KeyBinds, ViewHotkeys
from ..call_spots import view_codes, view_bleed_matrix, view_bled_codes, view_spot, view_intensity, gene_counts
from .. import call_spots as call_spots_plot
from ..call_spots_new import GEViewer, ViewBleedCalc, ViewAllGeneScores, BGNormViewer
from ..omp import view_omp, view_omp_fit, view_omp_score, histogram_score
from ..omp.coefs import view_score  # gives import error if call from call_spots.dot_product
from ... import call_spots
from ... import utils
from ...setup import Notebook
# set matplotlib background to dark
plt.style.use('dark_background')


class Viewer:
    def __init__(
        self,
        nb: Notebook,
        background_image: Optional[list] = ["dapi"],
        background_image_colour: Optional[list] = ["gray"],
        gene_marker_file: Optional[str] = None
    ) -> None:
        """
        This is the function to view the results of the pipeline i.e. the spots found and which genes they were
        assigned to.

        Args:
            nb: Notebook containing at least the `ref_spots` page.
            background_image: Optional list of file_names or images that will be plotted as the background image.
                If images, z dimensions need to be first i.e. `n_z x n_y x n_x` if 3D or `n_y x n_x` if 2D.
                If pass *2D* image for *3D* data, will show same image as background on each z-plane.
            background_image_colour: list of names of background colours. Must be same length as background_image
            gene_marker_file: Path to csv file containing marker and color for each gene. There must be 7 columns
                in the csv file with the following headers (comma separated):
                * ID - int, unique number for each gene, in ascending order
                * GeneNames - str, name of gene with first letter capital
                * ColorR - float, Rgb color for plotting
                * ColorG - float, rGb color for plotting
                * ColorB - float, rgB color for plotting
                * napari_symbol - str, symbol used to plot in napari
                * mpl_symbol - str, equivalent of napari symbol in matplotlib.
                If it is not provided, then the default file *coppafish/plot/results_viewer/legend.gene_color.csv*
                will be used.
        """
        # TODO: Make this more modular, write a function for preprocessing gene marker file
        self.is_3d = nb.basic_info.is_3d
        self.method_names = ["anchor", "prob"] + (["omp"] if nb.has_page("omp") else [])
        self.relevant_pages = ["ref_spots", "ref_spots", "omp"] if nb.has_page("omp") else ["ref_spots", "ref_spots"]

        # load in genes, spots and background images
        # declare variables
        self.genes = np.zeros((len(nb.call_spots.gene_names), 0)).tolist()
        self.spots = np.zeros((len(self.method_names), len(nb.call_spots.gene_names), 0)).tolist()
        self.background_image, self.background_image_colour, self.background_image_name = [], [], []
        # populate variables
        self.load_genes(gene_marker_file=gene_marker_file, nb_gene_names=nb.call_spots.gene_names)
        self.load_spots(nb=nb)
        self.load_bg_images(nb=nb, background_image=background_image, background_image_colour=background_image_colour)

        # create napari viewer
        self.viewer = napari.Viewer()
        self.format_napari_viewer()

        self.viewer_status_on_select()

        napari.run()

    # function to create the napari viewer
    def format_napari_viewer(self) -> None:
        """
        Create the napari viewer.
        """
        # Set up napari viewer

        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        self.z_thick = 1  # show +/- 1 plane initially
        self.nz = nb.basic_info.nz



        # Add all spots in layer as transparent white spots.
        self.point_size = [self.z_thick, 10, 10]  # with size=4, spots are too small to see
        self.viewer.add_points(
            self.spot_zyx,
            name="Diagnostic",
            face_color="w",
            size=np.array(self.point_size) + np.array([0, 2, 2]),
            opacity=0,
            shown=show_spots,
            out_of_slice_display=True,
        )

        if self.is_3d:
            self.z_thick_slider = QSlider(Qt.Orientation.Horizontal)
            self.z_thick_slider.setRange(0, self.nz)
            self.z_thick_slider.setValue(self.z_thick)
            self.z_thick_slider.valueChanged.connect(lambda x: self.change_z_thick(x))
            self.viewer.window.add_dock_widget(self.z_thick_slider, area="left", name="Z Thickness")
        config = self.nb.get_config()["thresholds"]
        self.score_omp_multiplier = config["score_omp_multiplier"]
        self.score_thresh_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)  # Slider to change score_thresh
        # Scores for anchor/omp are different so reset score range when change method
        # Max possible score is that found for ref_spots, as this can be more than 1.
        # Max possible omp score is 1.
        max_score = np.around(utils.round_any(nb.ref_spots.score.max(), 0.1, "ceil"), 2)
        max_score = float(np.clip(max_score, 1, np.inf))
        self.score_range = {"anchor": [config["score_ref"], max_score], "prob": [config["score_ref"], max_score]}
        if self.nb.has_page("omp"):
            self.score_range["omp"] = [config["score_omp"], max_score]
            self.score_thresh_slider.setValue(self.score_range["omp"])
        else:
            self.score_thresh_slider.setValue(self.score_range["anchor"])
        self.score_thresh_slider.setRange(0, max_score)
        # When dragging, status will show thresh.
        self.score_thresh_slider.valueChanged.connect(lambda x: self.show_score_thresh(x[0], x[1]))
        # On release of slider, genes shown will change
        self.score_thresh_slider.sliderReleased.connect(self.update_plot)
        self.viewer.window.add_dock_widget(self.score_thresh_slider, area="left", name="Score Range")

        # intensity is calculated same way for anchor / omp method so do not reset intensity threshold when change
        # method.
        self.intensity_thresh_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.intensity_thresh_slider.setRange(0, 1)
        intensity_thresh = call_spots.qual_check.get_intensity_thresh(nb)
        self.intensity_thresh_slider.setValue(intensity_thresh)
        # When dragging, status will show thresh.
        self.intensity_thresh_slider.valueChanged.connect(lambda x: self.show_intensity_thresh(x))
        # On release of slider, genes shown will change
        self.intensity_thresh_slider.sliderReleased.connect(self.update_plot)
        self.viewer.window.add_dock_widget(self.intensity_thresh_slider, area="left", name="Intensity Threshold")

        if self.nb.has_page("omp"):
            self.method_buttons = ButtonMethodWindow("OMP")  # Buttons to change between Anchor and OMP spots showing.
            self.method_buttons.button_omp.clicked.connect(self.button_omp_clicked)
        else:
            self.method_buttons = ButtonMethodWindow("Anchor", has_omp=False)
        # What does the below do?
        # This part of the code makes each button call a different function
        self.method_buttons.button_anchor.clicked.connect(self.button_anchor_clicked)
        self.method_buttons.button_prob.clicked.connect(self.button_prob_clicked)
        self.viewer.window.add_dock_widget(self.method_buttons, area="left", name="Method")

        self.key_call_functions()
        if self.nb.basic_info.is_3d:
            self.viewer.dims.axis_labels = ["z", "y", "x"]
        else:
            self.viewer.dims.axis_labels = ["y", "x"]

    # legend functions
    def add_legend(self, gene_legend_info: pd.DataFrame, cell_legend_info: pd.DataFrame) -> None:
        # Add legend indicating genes plotted
        self.legend = {"fig": None, "ax": None}
        self.legend["fig"], self.legend["ax"], n_gene_label_letters = legend.add_legend(
            gene_legend_info=gene_legend_info,
            genes=self.gene_names,
        )
        # xy is position of each symbol in legend, need to see which gene clicked on.
        self.legend["xy"] = np.zeros((len(self.legend["ax"].collections), 2), dtype=float)
        self.legend["gene_no"] = np.zeros(len(self.legend["ax"].collections), dtype=int)
        # In legend, each gene name label has at most n_gene_label_letters letters so need to crop
        # gene_names in notebook when looking for corresponding gene in legend.
        gene_names_crop = np.asarray([gene_name[:n_gene_label_letters] for gene_name in self.gene_names])
        for i in range(self.legend["xy"].shape[0]):
            # Position of label for each gene in legend window
            self.legend["xy"][i] = np.asarray(self.legend["ax"].collections[i].get_offsets())
            # gene no in notebook that each position in the legend corresponds to
            self.legend["gene_no"][i] = np.where(gene_names_crop == self.legend["ax"].texts[i].get_text())[0][0]
        self.legend["fig"].mpl_connect("button_press_event", self.update_genes)
        self.viewer.window.add_dock_widget(self.legend["fig"], area="left", name="Genes")
        self.active_genes = np.arange(len(self.gene_names))  # start with all genes shown

    # add some functions to load the data for the __init__ function
    def load_genes(self, gene_marker_file: str, nb_gene_names: list) -> None:
        """
        Load genes from the legend into the Viewer Object.
        Args:
            gene_marker_file: str, path to csv file containing marker and color for each gene. There must be 7 columns
                in the csv file with the following headers (comma separated):
                * ID - int, unique number for each gene, in ascending order
                * GeneNames - str, name of gene with first letter capital
                * ColorR - float, Rgb color for plotting
                * ColorG - float, rGb color for plotting
                * ColorB - float, rgB color for plotting
                * napari_symbol - str, symbol used to plot in napari
                * mpl_symbol - str, equivalent of napari symbol in matplotlib.
            nb_gene_names: list of gene names in the notebook
        """
        if gene_marker_file is None:
            gene_marker_file = importlib_resources.files("coppafish.plot.results_viewer").joinpath("gene_color.csv")
        gene_legend_info = pd.read_csv(gene_marker_file)
        legend_gene_names = gene_legend_info["GeneNames"]
        genes = []
        for i, g in enumerate(nb_gene_names):
            if g in legend_gene_names:
                colour = gene_legend_info[gene_legend_info["GeneNames"] == g][["ColorR", "ColorG", "ColorB"]].values[0]
                symbol_napari = gene_legend_info[gene_legend_info["GeneNames"] == g]["napari_symbol"].values[0]
                symbol_mpl = gene_legend_info[gene_legend_info["GeneNames"] == g]["mpl_symbol"].values[0]
                genes.append(Gene(name=g, notebook_index=i, colour=colour, symbol_mpl=symbol_mpl,
                                  symbol_napari=symbol_napari))
            else:
                genes.append(Gene(name=g, notebook_index=i, colour=None, symbol_mpl=None, symbol_napari=None))
        invisible_genes = [g for g in genes if g.symbol_mpl is None]
        if invisible_genes:
            warnings.warn(f"Genes {invisible_genes} are not in the gene marker file and will not be plotted.")
        self.genes = genes

    def load_bg_images(self, background_image: list, background_image_colour: list, nb: Notebook) -> None:
        """
        Load background images into the napari viewer and into the Viewer Object.
        Note: napari viewer must be created before calling this function.

        Args:
            background_image: list of file_names or images that will be plotted as the background image.
            background_image_colour: list of names of background colours. Must be same length as background_image
            nb: Notebook.

        """
        # make things lists if not already
        if not isinstance(background_image, list):
            background_image = [background_image]
        if not isinstance(background_image_colour, list):
            background_image_colour = [background_image_colour]
        background_image_dir = []

        # load images
        # users were allowed to specify a list of strings or images. If a string, assume it is a file name and try to
        # load it. Save file names in background_image_dir and images in background_image.
        for i, b in enumerate(background_image):
            # Check if b is a string, if so, assume it is a file name and try to load it. Then load the image into
            # background_image. If it is not a string, assume it is an image and load it directly into background_image.
            if not isinstance(b, str):
                background_image_dir.append(None)
                continue
            # If we have got this far, b is a string. Check if it is a keyword for a standard image.
            if b.lower() == "dapi":
                file_name = nb.file_names.big_dapi_image
            elif b.lower() == "anchor":
                file_name = nb.file_names.big_anchor_image
            else:
                file_name = b
            background_image_dir.append(file_name)
            # Now try to load the image
            if file_name is not None and os.path.isfile(file_name):
                if file_name.endswith(".npz"):
                    # Assume image is first array if .npz file. Now replace the string with the actual image.
                    background_image[i] = np.load(file_name)
                    background_image[i] = background_image[i].f.arr_0
                elif file_name.endswith(".npy"):
                    # Assume image is first array if .npz file. Now replace the string with the actual image.
                    background_image[i] = np.load(file_name)
                elif file_name.endswith(".tif"):
                    background_image[i] = skimage.io.imread(file_name)
            else:
                background_image[i] = None
                warnings.warn(
                    f"No file exists with file name =\n\t{file_name}\n so will not be plotted."
                )
            # Check if background image is constant. If so, don't plot it.
            if background_image[i] is not None and np.allclose(
                    [background_image[i].max()], [background_image[i].min()]
            ):
                warnings.warn(
                    f"Background image with file name =\n\t{file_name}"
                    + "\ncontains constant values, so not plotting"
                )
                background_image[i] = None
        # remove none entries from bg_images
        good = [i for i, b in enumerate(background_image) if b is not None]
        self.background_image = [background_image[i] for i in good]
        self.background_image_colour = [background_image_colour[i] for i in good]
        self.background_image_name = [os.path.basename(background_image_dir[i]) for i in good]

    def load_spots(self, nb: Notebook) -> None:
        """
        Load spots into the viewer and into the Viewer Object.
        Args:
            nb: Notebook containing at least the `call_spots` page.

        """
        # define frequently used variables
        n_methods = len(self.method_names)
        n_genes = len(self.gene_names)
        tile_origin = nb.stitch.tile_origin
        # initialise list of spots and load relevant information
        spots = np.zeros((len(self.method_names), len(self.gene_names), 0)).tolist()
        tile = [nb.__getattribute__(page).tile for page in self.relevant_pages]
        global_loc = [(nb.__getattribute__(self.relevant_pages[i]).local_yxz + tile_origin[tile[i]])[:, [2, 0, 1]]
                      for i in range(n_methods)]
        colours = [nb.__getattribute__(page).colors for page in self.relevant_pages]
        score = [nb.ref_spots.score, np.max(nb.ref_spots.gene_probs, axis=1)]
        gene_no = [nb.ref_spots.gene_no, np.argmax(nb.ref_spots.gene_probs, axis=1)]
        if nb.has_page("omp"):
            score.append(nb.omp.score)
            gene_no.append(nb.omp.gene_no)

        # add spots to Viewer object and the napari viewer
        for m, g in np.ndindex(n_methods, n_genes):
            mask = gene_no[m] == g
            indices = np.where(mask)[0]
            spots[m][g] = Spots(location=global_loc[m][mask], score=score[m][mask], tile=tile[m][mask], index=indices,
                                colours=colours[m][mask])
        self.spots = spots

    # Functions which will add data to the napari viewer
    def add_bg_images_to_viewer(self) -> None:
        """
        Add background images to the napari viewer (they already exist in the Viewer object).
        """
        # add images to viewer
        n_bg = len(self.background_image)
        self.image_contrast_slider = list(np.repeat(0, n_bg))

        # Loop through all background images and add them to the viewer.
        for i, b in enumerate(self.background_image):

            self.viewer.add_image(b, blending="additive", colormap=self.background_image_colour[i],
                                  name=self.background_image_name[i])
            self.viewer.layers[i].contrast_limits_range = [b.min(), b.max()]
            self.image_contrast_slider[i] = QRangeSlider(Qt.Orientation.Horizontal)
            self.image_contrast_slider[i].setRange(b.min(), b.max())
            # Make starting lower bound contrast the 95th percentile value so most appears black
            # Use mid_z to quicken calculation
            mid_z = int(b.shape[0] / 2)
            start_contrast = np.percentile(b[mid_z], [95, 99.9]).astype(int).tolist()
            self.image_contrast_slider[i].setValue(start_contrast)
            self.change_image_contrast(i)
            # When dragging, status will show contrast values.
            self.image_contrast_slider[i].valueChanged.connect(lambda x: self.show_image_contrast(x[0], x[1]))
            # On release of slider, genes shown will change
            self.image_contrast_slider[i].sliderReleased.connect(lambda j=i: self.change_image_contrast(i=j))
            # add slider to viewer
            self.viewer.window.add_dock_widget(self.image_contrast_slider[i],
                                               area="left", name=f"{self.background_image_name} Contrast")

    def viewer_status_on_select(self):
        # indicate selected data in viewer status.

        def indicate_selected(selectedData):
            if selectedData is not None:
                n_selected = len(selectedData)
                if n_selected == 1:
                    spot_no = list(selectedData)[0]
                    if self.method_buttons.method == "OMP":
                        spot_no = spot_no - self.omp_0_ind * 2
                        spot_gene = self.gene_names[self.nb.omp.gene_no[spot_no]]
                        tile = self.nb.omp.tile[spot_no]
                    elif self.method_buttons.method == "Anchor":
                        spot_gene = self.gene_names[self.nb.ref_spots.gene_no[spot_no]]
                        score = self.nb.ref_spots.score[spot_no]
                        tile = self.nb.ref_spots.tile[spot_no]
                    elif self.method_buttons.method == "Prob":
                        spot_no = spot_no % self.omp_0_ind
                        spot_gene = self.gene_names[np.argmax(self.nb.ref_spots.gene_probs, axis=1)[spot_no]]
                        score = np.max(self.nb.ref_spots.gene_probs[spot_no], axis=1)
                        tile = self.nb.ref_spots.tile[spot_no]

                    # OMP has no score so don't show it
                    if self.method_buttons.method == "OMP":
                        self.viewer.status = f"Spot {spot_no}, Gene {spot_gene}, Tile {tile} selected"
                    else:
                        self.viewer.status = (
                            f"Spot {spot_no}, Gene {spot_gene}, Score {score:.2f}, " f"Tile {tile} selected"
                        )
                elif n_selected > 1:
                    self.viewer.status = f"{n_selected} spots selected"

        """
        Listen to selected data changes
        """

        @napari.qt.thread_worker(connect={"yielded": indicate_selected})
        def _watchSelectedData(pointsLayer):
            selectedData = None
            while True:
                time.sleep(1 / 10)
                oldSelectedData = selectedData
                selectedData = pointsLayer.selected_data
                if oldSelectedData != selectedData:
                    yield selectedData
                yield None

        return _watchSelectedData(self.viewer.layers[self.transparent_spots_ind])

    def update_plot(self):
        # This updates the spots plotted to reflect score_range and intensity threshold selected by sliders,
        # method selected by button and genes selected through clicking on the legend.
        if self.method_buttons.method == "OMP":
            score = call_spots.qual_check.omp_spot_score(self.nb.omp)
            method_ind = np.arange(self.omp_0_ind * 2, self.n_spots)
            intensity_ok = self.nb.omp.intensity > self.intensity_thresh_slider.value()
        elif self.method_buttons.method == "Anchor":
            score = self.nb.ref_spots.score
            method_ind = np.arange(self.omp_0_ind)
            intensity_ok = self.nb.ref_spots.intensity > self.intensity_thresh_slider.value()
        elif self.method_buttons.method == "Prob":
            # Take the maximum gene probability as the score
            score = np.max(self.nb.ref_spots.gene_probs, axis=1)
            method_ind = np.arange(self.omp_0_ind, self.omp_0_ind * 2)
            # Ignore intensity slider
            intensity_ok = np.full(score.size, fill_value=True, dtype=bool)
        # Keep record of last score range set for each method
        self.score_range[self.method_buttons.method.lower()] = self.score_thresh_slider.value()
        qual_ok = np.array(
            [score > self.score_thresh_slider.value()[0], score <= self.score_thresh_slider.value()[1], intensity_ok]
        ).all(axis=0)
        spots_shown = np.zeros(self.n_spots, dtype=bool)
        # Only show spots which belong to a gene that is active and that passes quality threshold
        genes_shown = np.isin(self.spot_gene_no[method_ind], self.active_genes)
        spots_shown[method_ind[genes_shown]] = qual_ok[genes_shown]
        for i in range(len(self.viewer.layers)):
            if i == self.transparent_spots_ind:
                self.viewer.layers[i].shown = spots_shown
            elif self.label_prefix in self.viewer.layers[i].name:
                s = self.viewer.layers[i].name.replace(self.label_prefix, "")
                correct_gene = np.arange(len(self.nb.call_spots.gene_names))[self.gene_symbol == s]
                correct_spot = np.isin(self.spot_gene_no, correct_gene)
                self.viewer.layers[i].shown = spots_shown[correct_spot]

    def update_genes(self, event):
        # When click on a gene in the legend will remove/add that gene to plot.
        # When right-click on a gene, it will only show that gene.
        # When click on a gene which is the only selected gene, it will return to showing all genes.
        xy_clicked = np.array([event.xdata, event.ydata])
        xy_gene = np.zeros(2)
        for i in range(2):
            xy_gene[i] = self.legend["xy"][np.abs(xy_clicked[i] - self.legend["xy"][:, i]).argmin(), i]
        gene_clicked = np.where((self.legend["xy"] == xy_gene).all(axis=1))[0][0]
        gene_no = self.legend["gene_no"][gene_clicked]
        n_active = self.active_genes.size
        is_active = np.isin(gene_no, self.active_genes)
        active_genes_last = self.active_genes.copy()
        if is_active and n_active == 1:
            # If gene is only one selected, any click on it will return to all genes
            self.active_genes = np.sort(self.legend["gene_no"])
            # 1st argument in setdiff1d is always the larger array
            changed_genes = np.setdiff1d(self.active_genes, active_genes_last)
        elif event.button.name == "RIGHT":
            # If right-click on a gene, will select only that gene
            self.active_genes = np.asarray([gene_no])
            # 1st argument in setdiff1d is always the larger array
            changed_genes = np.setdiff1d(active_genes_last, self.active_genes)
            if not is_active:
                # also need to changed clicked gene if was not already active
                changed_genes = np.append(changed_genes, gene_no)
        elif is_active:
            # If single-click on a gene which is selected, will remove that gene
            self.active_genes = np.setdiff1d(self.active_genes, gene_no)
            changed_genes = np.asarray([gene_no])
        elif not is_active:
            # If single-click on a gene which is not selected, it will be removed
            self.active_genes = np.append(self.active_genes, gene_no)
            changed_genes = np.asarray([gene_no])

        # Change formatting
        for g in changed_genes:
            i = np.where(self.legend["gene_no"] == g)[0][0]
            if np.isin(g, self.active_genes):
                alpha = 1
            else:
                alpha = 0.5  # If not selected, make transparent
            self.legend["ax"].collections[i].set_alpha(alpha)
            self.legend["ax"].texts[i].set_alpha(alpha)
        self.legend["fig"].draw()
        self.update_plot()

    def show_score_thresh(self, low_value, high_value):
        self.viewer.status = self.method_buttons.method + ": Score Range = [{:.2f}, {:.2f}]".format(
            low_value, high_value
        )

    def show_intensity_thresh(self, value):
        self.viewer.status = "Intensity Threshold = {:.3f}".format(value)

    def show_image_contrast(self, low_value, high_value):
        # Show contrast of background image while dragging
        self.viewer.status = "Image Contrast Limits: [{:.0f}, {:.0f}]".format(low_value, high_value)

    def change_image_contrast(self, i):
        # Change contrast of background image
        self.viewer.layers[i].contrast_limits = [
            self.image_contrast_slider[i].value()[0],
            self.image_contrast_slider[i].value()[1],
        ]

    def change_z_thick(self, z_thick):
        # Show spots from different z-planes
        self.viewer.status = f"Z-thickness = {z_thick}"
        for i in range(len(self.viewer.layers)):
            self.viewer.layers[i].size = [z_thick, 10, 10]

    def button_anchor_clicked(self):
        self.method_buttons.button_anchor.setChecked(True)
        self.method_buttons.button_prob.setChecked(False)
        self.method_buttons.button_omp.setChecked(False)

        if self.method_buttons.method != "Anchor":
            self.method_buttons.method = "Anchor"
            # Because method has changed, also need to change score range
            self.score_thresh_slider.setValue(self.score_range["anchor"])
            self.update_plot()

    def button_prob_clicked(self):
        self.method_buttons.button_anchor.setChecked(False)
        self.method_buttons.button_prob.setChecked(True)
        self.method_buttons.button_omp.setChecked(False)

        if self.method_buttons.method != "Prob":
            self.method_buttons.method = "Prob"
            # Because method has changed, also need to change score range
            self.score_thresh_slider.setValue(self.score_range["prob"])
            self.update_plot()

    def button_omp_clicked(self):
        self.method_buttons.button_anchor.setChecked(False)
        self.method_buttons.button_prob.setChecked(False)
        self.method_buttons.button_omp.setChecked(True)

        if self.method_buttons.method != "OMP":
            # Logic on OMP button switch
            self.method_buttons.method = "OMP"
            # Because method has changed, also need to change score range
            self.score_thresh_slider.setValue(self.score_range["omp"])
            self.update_plot()

    def get_selected_spot(self):
        # Returns spot_no selected if only one selected (this is the spot_no relavent to the Notebook i.e.
        # if omp, the index of the spot in nb.omp is returned).
        # Otherwise, returns None and indicates why in viewer status.
        n_selected = len(self.viewer.layers[self.transparent_spots_ind].selected_data)
        if n_selected == 1:
            spot_no = list(self.viewer.layers[self.transparent_spots_ind].selected_data)[0]
            if self.method_buttons.method == "OMP":
                spot_no = spot_no - self.omp_0_ind * 2  # return spot_no as saved in self.nb for current method.
            else:
                spot_no = spot_no % self.omp_0_ind
        elif n_selected > 1:
            self.viewer.status = f"{n_selected} spots selected - need 1 to run diagnostic"
            spot_no = None
        else:
            self.viewer.status = "No spot selected :("
            spot_no = None
        return spot_no

    def key_call_functions(self):
        # Contains all functions which can be called by pressing a key with napari viewer open
        @Points.bind_key(KeyBinds.switch_zoom_select, overwrite=True)
        def change_zoom_select_mode(layer):
            if layer.mode == Mode.PAN_ZOOM:
                layer.mode = Mode.SELECT
                self.viewer.help = "Mode: Select"
            elif layer.mode == Mode.SELECT:
                layer.mode = Mode.PAN_ZOOM
                self.viewer.help = "Mode: Pan/Zoom"

        @self.viewer.bind_key(KeyBinds.view_hotkeys)
        def view_hotkeys(viewer):
            # Show Viewer keybindings
            ViewHotkeys()

        @self.viewer.bind_key(KeyBinds.remove_background)
        def remove_background_image(viewer):
            # Make background image visible / remove it
            if self.image_layer_ind is not None:
                for i in self.image_layer_ind:
                    if viewer.layers[i].visible:
                        viewer.layers[i].visible = False
                    else:
                        viewer.layers[i].visible = True

        @self.viewer.bind_key(KeyBinds.view_bleed_matrix)
        def call_to_view_bm(viewer):
            view_bleed_matrix(self.nb)

        @self.viewer.bind_key(KeyBinds.view_background_norm)
        def call_to_view_bg_norm(viewer):
            BGNormViewer(self.nb)

        @self.viewer.bind_key(KeyBinds.view_bleed_matrix_calculation)
        def call_to_view_bm_calc(viewer):
            ViewBleedCalc(self.nb)

        @self.viewer.bind_key(KeyBinds.view_bled_codes)
        def call_to_view_bm(viewer):
            view_bled_codes(self.nb)

        @self.viewer.bind_key(KeyBinds.view_all_gene_scores)
        def call_to_view_all_hists(viewer):
            ViewAllGeneScores(self.nb)

        @self.viewer.bind_key(KeyBinds.view_gene_efficiency)
        def call_to_view_gene_efficiency(viewer):
            GEViewer(self.nb)

        @self.viewer.bind_key(KeyBinds.view_gene_counts)
        def call_to_gene_counts(viewer):
            if self.nb.has_page("omp"):
                score_omp_thresh = self.score_range["omp"][0]
            else:
                score_omp_thresh = None
            score_thresh = self.score_range["anchor"][0]
            intensity_thresh = self.intensity_thresh_slider.value()
            gene_counts(self.nb, None, None, score_thresh, intensity_thresh, score_omp_thresh)

        @self.viewer.bind_key(KeyBinds.view_histogram_scores)
        def call_to_view_omp_score(viewer):
            if self.nb.has_page("omp"):
                score_multiplier = self.omp_score_multiplier_slider.value()
            else:
                score_multiplier = None
            histogram_score(self.nb, self.method_buttons.method, score_multiplier)

        @self.viewer.bind_key(KeyBinds.view_scaled_k_means)
        def call_to_view_omp_score(viewer):
            call_spots_plot.view_scaled_k_means(self.nb)

        @self.viewer.bind_key(KeyBinds.view_colour_and_codes)
        def call_to_view_codes(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_codes(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key(KeyBinds.view_spot_intensities)
        def call_to_view_spot(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_spot(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key(KeyBinds.view_spot_colours_and_weights)
        def call_to_view_omp_score(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_score(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key(KeyBinds.view_intensity_from_colour)
        def call_to_view_omp_score(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_intensity(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key(KeyBinds.view_omp_coefficients)
        def call_to_view_omp(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_omp(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key(KeyBinds.view_omp_fit)
        def call_to_view_omp(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_omp_fit(self.nb, spot_no, self.method_buttons.method)

        @self.viewer.bind_key(KeyBinds.view_omp_score)
        def call_to_view_omp_score(viewer):
            spot_no = self.get_selected_spot()
            if spot_no is not None:
                view_omp_score(self.nb, spot_no, self.method_buttons.method, self.omp_score_multiplier_slider.value())


class ButtonMethodWindow(QMainWindow):
    def __init__(self, active_button: str = "Anchor", has_omp: bool = True):
        super().__init__()
        self.button_prob = QPushButton("Prob", self)
        self.button_prob.setCheckable(True)
        self.button_prob.setGeometry(50, 2, 50, 28)  # left, top, width, height

        self.button_anchor = QPushButton("Anchor", self)
        self.button_anchor.setCheckable(True)
        self.button_anchor.setGeometry(105, 2, 50, 28)  # left, top, width, height

        self.button_omp = QPushButton("OMP", self)
        self.button_omp.setCheckable(has_omp)
        self.button_omp.setGeometry(160, 2, 50, 28)  # left, top, width, height

        if active_button.lower() == "omp" and has_omp:
            # Initially, show OMP spots
            self.button_omp.setChecked(True)
            self.method = "OMP"
        elif active_button.lower() == "anchor":
            # Initially, show Anchor spots
            self.button_anchor.setChecked(True)
            self.method = "Anchor"
        elif active_button.lower() == "prob":
            # Show gene probabilities
            self.button_prob.setChecked(True)
            self.method = "Prob"
        else:
            raise ValueError(f"Unexpected active_button, {active_button}, was given.")


class Spots:
    """
    Class to hold different spot information. In the Viewer class we will have a list of lists of Spots objects, one
    for each gene within each method.
    """
    def __init__(self, location: np.ndarray, colours: np.ndarray, score: np.ndarray, tile: np.ndarray,
                 index: np.ndarray):
        """
        Create object for spots of a single gene within a single method.
        Args:
            location: (np.ndarray) of shape (n_spots, 3) with the zyx location of each spot. (int16)
            colours: (np.ndarray) of shape (n_spots, n_rounds, n_channels) of the raw colour of each spot. (uint16)
            score: (np.ndarray) of shape (n_spots,) with the score of each spot. (float32)
            tile: (np.ndarray) of shape (n_spots,) with the tile of each spot. (int16)
            index: (np.ndarray) of shape (n_spots,) with the index of each spot within its method. (int16)
        """
        assert len(location) == len(colours) == len(score) == len(tile) == len(index) # Check all arrays are same length
        self.location = location
        self.colours = colours
        self.score = score
        self.tile = tile
        self.index = index


class Gene:
    """
    Class to hold gene information. In the Viewer class we will have a list of Gene objects, one for each gene.
    This will store the gene name, index, colour and symbol.
    """
    def __init__(self, name: str, notebook_index: int, colour: np.ndarray, symbol_mpl: str, symbol_napari: str):
        """
        Create object for a single gene.
        Args:
            name: (str) gene name.
            notebook_index: (int) index of the gene within the notebook.
            colour: (np.ndarray) of shape (3,) with the RGB colour of the gene. (int8)
            symbol_mpl: (str) symbol used to plot in matplotlib. (Used in the legend)
            symbol_napari: (str) symbol used to plot in napari. (Used in the viewer)
        """
        self.name = name
        self.notebook_index = notebook_index
        self.colour = colour
        self.symbol_mpl = symbol_mpl
        self.symbol_napari = symbol_napari