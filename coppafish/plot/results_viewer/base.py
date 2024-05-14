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
from typing import Optional, Union

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
        # declare variables needed from the notebook
        self.is_3d, self.nz, self.abs_intensity_thresh = (nb.basic_info.is_3d, nb.basic_info.nz,
                                                          nb.call_spots.abs_intensity_thresh[50])
        self.method_names = ["anchor", "prob"] + (["omp"] if nb.has_page("omp") else [])
        self.relevant_pages = ["ref_spots", "ref_spots", "omp"] if nb.has_page("omp") else ["ref_spots", "ref_spots"]

        # declare image and spot variables
        self.genes, self.spots = None, None
        self.background_image, self.background_image_colour, self.background_image_name = [], [], []
        # populate variables
        self.create_gene_list(gene_marker_file=gene_marker_file, nb_gene_names=nb.call_spots.gene_names)
        self.create_spots_list(nb=nb)
        self.load_bg_images(nb=nb, background_image=background_image, background_image_colour=background_image_colour)

        # create napari viewer
        self.viewer = napari.Viewer()
        self.legend = {}
        self.image_contrast_slider = []
        self.score_range_slider, self.intensity_thresh_slider = None, None
        if self.is_3d:
            self.z_thick = 1
            self.z_thick_slider = None
        self.method_buttons = None
        self.format_napari_viewer(gene_marker_file=gene_marker_file)

        # add images and spots to napari viewer. Hook up the selection status to the viewer status, and key bindings
        self.active_method = 2 if nb.has_page("omp") else 0
        self.active_genes = np.arange(len(nb.call_spots.gene_names))
        self.add_data_to_viewer()
        self.update_status_continually()
        self.key_call_functions()
        napari.run()

    def format_napari_viewer(self, gene_marker_file: pd.DataFrame) -> None:
        """
        Create the napari viewer.
        """
        # turn off layer list and layer controls
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        # add gene legend to viewer
        self.add_legend(gene_legend_info=gene_marker_file)

        # Z-thickness slider
        if self.is_3d:
            self.add_slider(name="z_thick", value=1, slider_range=(0, self.nz), slider_mode="single",
                            slider_variable="z_thick")

        # set up sliders to adjust background images contrast
        for i, b in enumerate(self.background_image):
            if b.ndim == 3:
                mid_z = int(b.shape[0] / 2)
                start_contrast = np.percentile(b[mid_z], [95, 99]).astype(int).tolist()
            else:
                start_contrast = np.percentile(b, [95, 99]).astype(int).tolist()
            self.add_slider(name=self.background_image_name[i], value=start_contrast, slider_range=(b.min(), b.max()),
                            slider_mode="range", slider_variable="image")

        # Slider to change score_thresh
        self.add_slider(name="score_thresh", value=(0.3, 1), slider_range=(0, 1), slider_mode="range",
                        slider_variable="spot")
        # Slider to change intensity_thresh
        self.add_slider(name="intensity_thresh", value=self.abs_intensity_thresh, slider_range=(0, 1),
                        slider_mode="single", slider_variable="spot")
        # Buttons to change method
        self.method_buttons = Method(active_button=self.method_names[self.active_method],
                                     has_omp=self.method_names[-1] == "omp")
        for i in range(len(self.method_names)):
            self.method_buttons.button[i].clicked.connect(lambda x=i: self.method_event_handler(method=x))
            self.method_buttons.button[i].clicked.connect(self.update_layers)
        self.viewer.window.add_dock_widget(self.method_buttons, area="left", name="Method")

        # connect a change in z-plane to update the z-thick slider
        self.viewer.dims.events.current_step.connect(self.update_z_thick)

        # label the axes
        if self.is_3d:
            self.viewer.dims.axis_labels = ["z", "y", "x"]
        else:
            self.viewer.dims.axis_labels = ["y", "x"]

    def add_legend(self, gene_legend_info: pd.DataFrame) -> None:
        # Add legend indicating genes plotted
        self.legend["fig"], self.legend["ax"], n_gene_label_letters = legend.add_legend(
            gene_legend_info=gene_legend_info,
            genes=self.gene_names,
        )

        # Initialize positions and gene numbers in the legend
        num_genes = len(self.legend["ax"].collections)
        self.legend["xy"] = np.zeros((num_genes, 2), dtype=float)
        self.legend["gene_no"] = np.zeros(num_genes, dtype=int)

        # Crop gene names to the maximum label length
        cropped_gene_names = np.array([gene[:max_label_length] for gene in self.gene_names])

        # Populate positions and gene numbers for the legend
        for idx in range(num_genes):
            self.legend["xy"][idx] = np.array(self.legend["ax"].collections[idx].get_offsets())
            legend_text = self.legend["ax"].texts[idx].get_text()
            self.legend["gene_no"][idx] = np.where(cropped_gene_names == legend_text)[0][0]

        # Connect the event handler for legend clicks
        self.legend["fig"].mpl_connect("button_press_event", self.legend_event_handler)

        # Add the legend figure to the viewer window
        self.viewer.window.add_dock_widget(self.legend["fig"], area="left", name="Genes")

    def add_slider(self, name: str, value: Union[float, tuple], slider_range: tuple,
                   slider_mode: str = "single", slider_variable: str = "spot") -> None:
        """
        Create a slider and add it to the napari viewer.
        Args:
            name: str, name of the slider.
            value: float, value of the slider. (if mode is "range", value should be a tuple of floats)
            slider_range: tuple, range of the slider.
            slider_mode: str, either "range" or "single" depending on whether the slider is a range slider or a
             single slider.
            slider_variable: str, either "spot", "image" or "z_thick" depending on what the slider is controlling.
        """
        assert slider_mode in ["range", "single"], "mode must be either 'range' or 'single'"
        assert slider_variable in ["spot", "image", "z_thick"], "variable must be either 'spot' or 'image'"

        if slider_mode == "range":
            slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        else:
            slider = QDoubleSlider(Qt.Orientation.Horizontal)
        slider.setRange(slider_range[0], slider_range[1])
        slider.setValue(value)

        # connect to show_slider_value to show user the new value of the slider
        slider.valueChanged.connect(lambda x: self.show_slider_value(
            string=f'{self.method_names[self.active_method]} {name}', value=np.array(x)))

        # connect to the appropriate function to update the napari viewer
        if slider_variable == "spot":
            slider.sliderReleased.connect(self.update_thresholds)
            self.__setattr__(f"{name}_slider", slider)
        elif slider_variable == "image":
            layer_ind = self.background_image_name.index(name)
            slider.sliderReleased.connect(lambda i=layer_ind: self.change_image_contrast(i))
            self.image_contrast_slider.append(slider)
        elif slider_variable == "z_thick":
            slider.sliderReleased.connect(self.update_z_thick)
            self.z_thick_slider = slider

        # add slider to napari viewer
        self.viewer.window.add_dock_widget(slider, area="left", name=name)

    def show_slider_value(self, string, value):
        """
        Show the value of a slider in the viewer status.
        Args:
            string: String to show before the value.
            value: np.ndarray, value of the slider.
        """
        self.viewer.status = f"{string}: {np.round(value, 2)}"

    def update_image_contrast(self, i):
        # Change contrast of background image
        self.viewer.layers[i].contrast_limits = [
            self.image_contrast_slider[i].value()[0],
            self.image_contrast_slider[i].value()[1],
        ]

    def update_layers(self) -> None:
        """
        This method updates the layers in the napari viewer to reflect the current state of the Viewer object. It will
        be called when the active method or active genes change.
        """
        # update active genes and methods in the transparent layer
        for m in range(len(self.method_names)):
            layer_name = f"{self.method_names[m]} all"
            self.viewer.layers[layer_name].visible = (m == self.active_method)

        # update active genes and methods in the non-transparent layers
        for m, g in np.ndindex(len(self.method_names), len(self.genes)):
            layer_name = f"{self.method_names[m]} {self.genes[g].name}"
            self.viewer.layers[layer_name].visible = (m == self.active_method and g in self.active_genes)

        # update active layer in napari viewer to the transparent all layer
        active_layer_name = f"{self.method_names[self.active_method]} all"
        self.viewer.layers.selection.active = self.viewer.layers[active_layer_name]

    def update_thresholds(self) -> None:
        """
        This method updates the thresholds in the napari viewer to reflect the current state of the Viewer object. It
        will be called when the score or intensity thresholds change.

        The only thresholds this method will update are the score range and intensity threshold.

        The z-thickness slider will be updated in the method `update_z_thick`.
        """
        score_range = self.score_thresh_slider.value()
        intensity_thresh = self.intensity_thresh_slider.value()
        # We add 1 to the genes as the last entry of the spots is all genes
        # Turn off spots by setting size to 0
        for m, g in np.ndindex(len(self.method_names), len(self.genes) + 1):
            layer_name = f"{self.method_names[m]} {self.genes[g].name}"
            # 1. score range
            good_score = (self.spots[m][g].score >= score_range[0]) & (self.spots[m][g].score <= score_range[1])
            # 2. intensity threshold
            good_intensity = self.spots[m][g].intensity >= intensity_thresh
            # create mask
            mask = good_score & good_intensity
            size = np.zeros(len(mask), dtype=int)
            size[mask] = 10
            self.viewer.layers[layer_name].size = size

    def update_z_thick(self) -> None:
        """
        This method updates the z-thickness in the napari viewer to reflect the current state of the Viewer object.
        It will be called when the z-thickness changes, or the z-position of the viewer changes.
        """
        z_thick = self.z_thick_slider.value()
        # we will change the z_thickness of the spots for each points layer. Do this by adaptively scaling and shifting
        # the z-coords of the spots. This will make the spots appear thicker in the z-direction.
        scale = 1 / (2 * z_thick + 1)
        shift = (1 - scale) * self.viewer.dims.current_step[0]
        for layer in self.viewer.layers:
            layer.scale = (scale, 1, 1)
            layer.translate = (shift, 0, 0)

        # napari automatically adjusts the size of the z-step when points are scaled, so we undo that below:
        v_range = list(self.viewer.dims.range)
        v_range[0] = (0, self.nz - 1, 1)
        self.viewer.dims.range = tuple(v_range)

    def update_status_continually(self):
        """
        Update the status of the viewer to show the number of spots selected, and if 1 spot is selected, show the gene,
        score and tile of that spot.
        """
        def indicate_selected(selectedData):
            if selectedData is not None:
                n_selected = len(selectedData)
                if n_selected == 1:
                    spot_index = list(selectedData)[0]
                    spot_gene = self.spots[self.active_method][-1][spot_index].gene
                    tile = self.spots[self.active_method][-1][spot_index].tile
                    score = self.spots[self.active_method][-1][spot_index].score
                    self.viewer.status = (
                        f"Spot {spot_index}, Gene {spot_gene}, Score {score:.2f}, " f"Tile {tile} selected"
                    )
                elif n_selected > 1:
                    self.viewer.status = f"{n_selected} spots selected"

        # This decorator will run the function in a separate thread, so that the napari viewer does not freeze.
        @napari.qt.thread_worker(connect={"yielded": indicate_selected})
        def _watchSelectedData(pointsLayer):
            """
                Listen to selected data changes
            """
            selectedData = None
            while True:
                time.sleep(1 / 10)
                oldSelectedData = selectedData
                selectedData = pointsLayer.selected_data
                if oldSelectedData != selectedData:
                    yield selectedData
                yield None

        return _watchSelectedData(self.viewer.layers[self.viewer.layers.selection.active])

    def method_event_handler(self, method: int):
        for i in range(3):
            self.method_buttons.button[i].setChecked(method == i)
        self.active_method = method
        # change the active layer to the transparent spots layer for the new method
        active_layer_name = f"{self.method_names[self.active_method]} all"
        self.viewer.layers.selection.active = self.viewer.layers[active_layer_name]
        self.update_layers()

    def legend_event_handler(self, event):
        """
            Update the genes plotted in the napari viewer based on the gene that was clicked in the legend.
            When click on a gene in the legend will remove/add that gene to plot.
            When right-click on a gene, it will only show that gene.
            When click on a gene which is the only selected gene, it will return to showing all genes.
        """
        clicked_coordinates = np.array([event.xdata, event.ydata])
        closest_gene_coordinates = np.zeros(2)

        # Find the closest gene coordinates in the legend to the clicked position
        for i in range(2):
            closest_grid_point = np.abs(clicked_coordinates[i] - self.legend["xy"][:, i]).argmin()
            closest_gene_coordinates[i] = self.legend["xy"][closest_grid_point, i]

        # Identify the gene that was clicked
        clicked_gene_index = np.where((self.legend["xy"] == closest_gene_coordinates).all(axis=1))[0][0]
        clicked_gene_number = self.legend["gene_no"][clicked_gene_index]

        # Get the current number of active genes and check if the clicked gene is active
        num_active_genes = self.active_genes.size
        is_gene_active = np.isin(clicked_gene_number, self.active_genes)
        previous_active_genes = self.active_genes.copy()

        # Update the active genes based on the click event
        if is_gene_active and num_active_genes == 1:
            # If the gene is the only selected gene, clicking it will show all genes
            self.active_genes = np.sort(self.legend["gene_no"])
            changed_genes = np.setdiff1d(self.active_genes, previous_active_genes)
        elif event.button.name == "RIGHT":
            # If right-clicking on a gene, it will select only that gene
            self.active_genes = np.array([clicked_gene_number])
            changed_genes = np.setdiff1d(previous_active_genes, self.active_genes)
            if not is_gene_active:
                # If the clicked gene was not active, add it to the changed genes
                changed_genes = np.append(changed_genes, clicked_gene_number)
        elif is_gene_active:
            # If single-clicking on an active gene, it will remove that gene
            self.active_genes = np.setdiff1d(self.active_genes, clicked_gene_number)
            changed_genes = np.array([clicked_gene_number])
        else:
            # If single-clicking on an inactive gene, it will add that gene
            self.active_genes = np.append(self.active_genes, clicked_gene_number)
            changed_genes = np.array([clicked_gene_number])

        # Update the legend formatting to reflect the changes
        for gene_number in changed_genes:
            gene_index = np.where(self.legend["gene_no"] == gene_number)[0][0]
            alpha_value = 1 if np.isin(gene_number, self.active_genes) else 0.5
            self.legend["ax"].collections[gene_index].set_alpha(alpha_value)
            self.legend["ax"].texts[gene_index].set_alpha(alpha_value)

        # Redraw the legend figure to apply the changes
        self.legend["fig"].draw()
        self.update_layers()

    def get_selected_spot_index(self):
        """
            Get the index of the selected spot.
        """
        n_selected = len(self.viewer.layers[self.viewer.layers.selection.active].selected_data)
        if n_selected == 1:
            spot_index = list(self.viewer.layers[self.viewer.layers.selection.active].selected_data)[0]
        elif n_selected > 1:
            self.viewer.status = f"{n_selected} spots selected - need 1 to run diagnostic"
            spot_index = None
        else:
            self.viewer.status = "No spot selected :("
            spot_index = None
        return spot_index

    def create_gene_list(self, gene_marker_file: str, nb_gene_names: list) -> None:
        """
        Create a list of genes from the notebook to store information about each gene. This will be saved to the object
         as self.genes. Each element of the list will be a Gene object. So it will contain the name, location,
         colour and symbols for each gene.
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

        # create a list of genes with the relevant information. If the gene is not in the gene marker file, it will be
        # added with None values for colour and symbols.
        for i, g in enumerate(nb_gene_names):
            if g in legend_gene_names:
                colour = gene_legend_info[gene_legend_info["GeneNames"] == g][["ColorR", "ColorG", "ColorB"]].values[0]
                symbol_napari = gene_legend_info[gene_legend_info["GeneNames"] == g]["napari_symbol"].values[0]
                symbol_mpl = gene_legend_info[gene_legend_info["GeneNames"] == g]["mpl_symbol"].values[0]
                genes.append(Gene(name=g, notebook_index=i, colour=colour, symbol_mpl=symbol_mpl,
                                  symbol_napari=symbol_napari))
            else:
                genes.append(Gene(name=g, notebook_index=i, colour=None, symbol_mpl=None, symbol_napari=None))

        # warn if any genes are not in the gene marker file
        invisible_genes = [g for g in genes if g.symbol_mpl is None]
        if invisible_genes:
            warnings.warn(f"Genes {invisible_genes} are not in the gene marker file and will not be plotted.")
        self.genes = genes

    def create_spots_list(self, nb: Notebook) -> None:
        """
        Create a list of spots from the notebook to store information about each spot. This will be saved to the object
        as self.spots. Each element of the list will be a Spots object. So it will contain the location, score, tile,
        colour and intensity for each spot.

        Note that self.spots is a list of lists. The first list is for each method, the second list is for each gene.
        The last element of the second list is for all genes.

        Args:
            nb: Notebook containing at least the `call_spots` page.

        """
        # define frequently used variables
        n_methods = len(self.method_names)
        n_genes = len(nb.gene_names)
        tile_origin = nb.stitch.tile_origin

        # initialise list of spots and load relevant information
        spots = np.zeros((len(self.method_names), len(self.gene_names) + 1, 0)).tolist()
        tile = [nb.__getattribute__(page).tile for page in self.relevant_pages]
        global_loc = [(nb.__getattribute__(self.relevant_pages[i]).local_yxz + tile_origin[tile[i]])[:, [2, 0, 1]]
                      for i in range(n_methods)]
        colours = [nb.__getattribute__(page).colors for page in self.relevant_pages]
        score = [nb.ref_spots.score, np.max(nb.ref_spots.gene_probs, axis=1)]
        gene_no = [nb.ref_spots.gene_no, np.argmax(nb.ref_spots.gene_probs, axis=1)]
        if nb.has_page("omp"):
            score.append(nb.omp.score)
            gene_no.append(nb.omp.gene_no)

        # add spots for each method and gene
        for m, g in np.ndindex(n_methods, n_genes):
            mask = gene_no[m] == g
            spots[m][g] = Spots(location=global_loc[m][mask],
                                score=score[m][mask],
                                tile=tile[m][mask],
                                colours=colours[m][mask],
                                intensity=nb.__getattribute__(self.relevant_pages[m]).intensity[mask],
                                gene=g)

        # add all spots for each method as the last element of the list
        for m in range(n_methods):
            spots[m][-1] = Spots(location=global_loc[m],
                                 score=score[m],
                                 tile=tile[m],
                                 colours=colours[m],
                                 intensity=nb.__getattribute__(self.relevant_pages[m]).intensity,
                                 gene=gene_no[m])
        self.spots = spots

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

    def add_data_to_viewer(self) -> None:
        """
        Add background images and spots to the napari viewer (they already exist in the Viewer object).
        """
        # Loop through all background images and add them to the viewer.
        for i, b in enumerate(self.background_image):
            self.viewer.add_image(b, blending="additive", colormap=self.background_image_colour[i],
                                  name=self.background_image_name[i])
            self.viewer.layers[i].contrast_limits_range = [b.min(), b.max()]

        # add a single layer of transparent spots to viewer for each method
        for m in range(len(self.method_names)):
            point_loc = self.spots[m][-1].location
            self.viewer.add_points(point_loc, size=10, face_color="white", name=f'{self.method_names[m]} all',
                                   opacity=0, out_of_slice_display=True)

        # add spots for each method and gene to viewer
        for m, g in np.ndindex(len(self.method_names), len(self.genes)):
            self.viewer.add_points(
                self.spots[m][g].location,
                size=10,
                face_color=self.spots[m][g].colours,
                symbol=self.genes[g].symbol_napari,
                name=f"{self.method_names[m]} {self.genes[g].name}",
                out_of_slice_display=True,
            )

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
            for i in range(len(self.background_image)):
                self.viewer.layers[self.background_image_name[i]].visible = not self.viewer.layers[
                    self.background_image_name[i]].visible

        @self.viewer.bind_key(KeyBinds.view_bleed_matrix)
        def call_to_view_bm(viewer):
            view_bleed_matrix(self.nb)

        @self.viewer.bind_key(KeyBinds.view_background_norm)
        def call_to_view_bg_norm(viewer):
            BGNormViewer(self.nb)

        # TODO: Remove or refactor this as this is not at all how we currently calculate bleed matrix
        # @self.viewer.bind_key(KeyBinds.view_bleed_matrix_calculation)
        # def call_to_view_bm_calc(viewer):
        #     ViewBleedCalc(self.nb)

        @self.viewer.bind_key(KeyBinds.view_bled_codes)
        def call_to_view_bm(viewer):
            view_bled_codes(self.nb)

        @self.viewer.bind_key(KeyBinds.view_all_gene_scores)
        def call_to_view_all_hists(viewer):
            ViewAllGeneScores(self.nb)

        @self.viewer.bind_key(KeyBinds.view_gene_efficiency)
        def call_to_view_gene_efficiency(viewer):
            GEViewer(self.nb)

        # # TODO: Remove or refactor this as this as we don't have a different score thresh for omp
        # @self.viewer.bind_key(KeyBinds.view_gene_counts)
        # def call_to_gene_counts(viewer):
        #     score_thresh = self.score_thresh_slider.value()[0]
        #     intensity_thresh = self.intensity_thresh_slider.value()
        #     gene_counts(self.nb, None, None, score_thresh, intensity_thresh, score_omp_thresh)
        #
        # @self.viewer.bind_key(KeyBinds.view_histogram_scores)
        # def call_to_view_omp_score(viewer):
        #     if self.nb.has_page("omp"):
        #         score_multiplier = self.omp_score_multiplier_slider.value()
        #     else:
        #         score_multiplier = None
        #     histogram_score(self.nb, self.method_buttons.method, score_multiplier)
        #
        # @self.viewer.bind_key(KeyBinds.view_scaled_k_means)
        # def call_to_view_omp_score(viewer):
        #     call_spots_plot.view_scaled_k_means(self.nb)

        @self.viewer.bind_key(KeyBinds.view_colour_and_codes)
        def call_to_view_codes(viewer):
            spot_index = self.get_selected_spot()
            if spot_index is not None:
                view_codes(self.nb, spot_index, self.method_names[self.active_method])

        @self.viewer.bind_key(KeyBinds.view_spot_intensities)
        def call_to_view_spot(viewer):
            spot_index = self.get_selected_spot()
            if spot_index is not None:
                view_spot(self.nb, spot_index, self.method_names[self.active_method])

        @self.viewer.bind_key(KeyBinds.view_spot_colours_and_weights)
        def call_to_view_omp_score(viewer):
            spot_index = self.get_selected_spot()
            if spot_index is not None:
                view_score(self.nb, spot_index, self.method_names[self.active_method])

        @self.viewer.bind_key(KeyBinds.view_intensity_from_colour)
        def call_to_view_omp_score(viewer):
            spot_index = self.get_selected_spot()
            if spot_index is not None:
                view_intensity(self.nb, spot_index, self.method_names[self.active_method])

        @self.viewer.bind_key(KeyBinds.view_omp_coefficients)
        def call_to_view_omp(viewer):
            spot_index = self.get_selected_spot()
            if spot_index is not None:
                view_omp(self.nb, spot_index, self.method_names[self.active_method])

        @self.viewer.bind_key(KeyBinds.view_omp_fit)
        def call_to_view_omp(viewer):
            spot_index = self.get_selected_spot()
            if spot_index is not None:
                view_omp_fit(self.nb, spot_index, self.method_names[self.active_method])

        # TODO: Remove or refactor this as this as we don't have a score multiplier for omp
        # @self.viewer.bind_key(KeyBinds.view_omp_score)
        # def call_to_view_omp_score(viewer):
        #     spot_index = self.get_selected_spot()
        #     if spot_index is not None:
        #         view_omp_score(self.nb, spot_index, self.method_buttons.method, self.omp_score_multiplier_slider.value())


class Method(QMainWindow):
    def __init__(self, active_button: str = "anchor", has_omp: bool = True):
        """
        Create a window with buttons to change between anchor, prob and omp spots. Will have the attributes:
        button_prob, button_anchor, button_omp (if has_omp is True), active_method.
        Args:
            active_button: (str) name of the button that should be active initially. Must be one of "anchor", "prob" or
                "omp" (if has_omp is True).
            has_omp: (bool) whether the notebook has an OMP page.
        """
        assert active_button in ["anchor", "prob", "omp"]
        assert has_omp or active_button != "omp"

        super().__init__()
        self.button = []

        self.button.append(QPushButton("anchor", self))
        self.button[-1].setCheckable(True)
        self.button[-1].setGeometry(50, 2, 50, 28)  # left, top, width, height

        self.button.append(QPushButton("prob", self))
        self.button[-1].setCheckable(True)
        self.button[-1].setGeometry(105, 2, 50, 28)  # left, top, width, height

        if has_omp:
            self.button.append(QPushButton("omp", self))
            self.button[-1].setCheckable(True)
            self.button[-1].setGeometry(160, 2, 50, 28)

        if active_button == "anchor":
            # Initially, show Anchor spots
            self.button[0].setChecked(True)
            self.active_method = 0
        elif active_button == "prob":
            # Show gene probabilities
            self.button[1].setChecked(True)
            self.active_method = 1
        elif active_button == "omp" and has_omp:
            # Initially, show OMP spots
            self.button[2].setChecked(True)
            self.active_method = 2


class Spots:
    """
    Class to hold different spot information. In the Viewer class we will have a list of lists of Spots objects, one
    for each gene within each method.
    """
    def __init__(self, location: np.ndarray, colours: np.ndarray, score: np.ndarray, tile: np.ndarray, intensity: np.ndarray,
                 gene: Union[int, np.ndarray]):
        """
        Create object for spots of a single gene within a single method.
        Args:
            location: (np.ndarray) of shape (n_spots, 3) with the zyx location of each spot. (int16)
            colours: (np.ndarray) of shape (n_spots, n_rounds, n_channels) of the raw colour of each spot. (uint16)
            score: (np.ndarray) of shape (n_spots,) with the score of each spot. (float32)
            tile: (np.ndarray) of shape (n_spots,) with the tile of each spot. (int16)
            intensity: (np.ndarray) of shape (n_spots,) with the intensity of each spot. (float32)
            gene: (int) gene number of the gene that the spots belong to. (int16)
                or (np.ndarray) of shape (n_spots,) with the gene number of each spot. (int16)
        """
        assert len(location) == len(colours) == len(score) == len(tile) == len(intensity)   # Check all same length
        if len(gene) > 1:
            assert len(location) == len(gene)
        self.location = location
        self.colours = colours
        self.score = score
        self.tile = tile
        self.intensity = intensity
        self.gene = gene


class Gene:
    """
    Class to hold gene information. In the Viewer class we will have a list of Gene objects, one for each gene.
    This will store the gene name, index, colour and symbol.
    """
    def __init__(self, name: str, notebook_index: int, colour: Union[np.ndarray, None], symbol_mpl: Union[str, None],
                 symbol_napari: Union[str, None]):
        """
        Create object for a single gene.
        Args:
            name: (str) gene name.
            notebook_index: (int) index of the gene within the notebook.
            colour: (np.ndarray) of shape (3,) with the RGB colour of the gene. (int8) or None (if not in gene marker
                file).
            symbol_mpl: (str) symbol used to plot in matplotlib. (Used in the legend) or None (if not in gene marker
                file).
            symbol_napari: (str) symbol used to plot in napari. (Used in the viewer) or None (if not in gene marker
                file).
        """
        self.name = name
        self.notebook_index = notebook_index
        self.colour = colour
        self.symbol_mpl = symbol_mpl
        self.symbol_napari = symbol_napari