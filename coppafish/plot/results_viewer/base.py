import copy
import os
import time
from typing import Optional, Union
import warnings

import importlib.resources as importlib_resources
import matplotlib.pyplot as plt
import napari
from napari.layers.points import Points
from napari.layers.points._points_constants import Mode
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QPushButton
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider
import tifffile

from . import legend
from .. import call_spots as call_spots_plot
from ...omp import base as omp_base
from ...setup import Notebook
from ..call_spots import view_bled_codes, ViewBleedMatrix, view_codes, view_spot, ViewScalingAndBGRemoval, GeneEfficiencyViewer
from ..call_spots import ViewAllGeneHistograms, HistogramScore
from ..omp import ViewOMPImage, ViewOMPPixelColours
from .hotkeys import KeyBinds, ViewHotkeys



class Viewer:
    def __init__(
        self,
        nb: Notebook,
        background_image: Optional[list] = ["dapi"],
        background_image_colour: Optional[list] = ["gray"],
        background_image_max_intensity_projection: Optional[list] = [False],
        gene_marker_file: Optional[str] = None,
        spot_size: int = 10,
        downsample_factor: int = 1,
    ) -> None:
        """
        This is the function to view the results of the pipeline i.e. the spots found and which genes they were
        assigned to.

        The Viewer object will have the following attributes:
            * method: dict, containing method names used to assign genes to spots, their corresponding pages in the
                notebook and the names of the buttons in the napari viewer.
            * background_images: dict, containing the background images to be plotted in the napari viewer. The keys are
                'image', 'name' and 'colour'.
            * genes: list, containing information about each gene. Each element of the list will be a Gene object.
            * spots: dict, containing all spots for each method. The keys are the method names and the values are Spots
                objects.
            * viewer: napari.Viewer, the napari viewer object.
            * legend: dict, containing the legend figure and axes.
            * sliders: dict, containing the sliders in the napari viewer.
            * nb: Notebook, the notebook object.
        Args:
            nb: Notebook containing at least the `ref_spots` page.
            background_image: Optional list of file_names or images that will be plotted as the background image.
                If images, z dimensions need to be first i.e. `n_z x n_y x n_x` if 3D or `n_y x n_x` if 2D.
                If pass *2D* image for *3D* data, will show same image as background on each z-plane.
            background_image_colour: list of names of background colours. Must be same length as background_image
            background_image_max_intensity_projection: Optional list of bools. If True, will plot the maximum intensity
                projection of the background image. If False, will plot the background_image as is.
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
            spot_size: int, size of the spots to be plotted in the napari viewer.
            downsample_factor: int, factor by which to downsample the images in y and x. Default is 1 which means no
                downsampling.
        """
        # set matplotlib background to dark
        plt.style.use("dark_background")
        # set up gene legend info
        if gene_marker_file is None:
            gene_marker_file = importlib_resources.files("coppafish.plot.results_viewer").joinpath("gene_color.csv")
        gene_legend_info = pd.read_csv(gene_marker_file)

        # declare variables needed from the notebook
        self.nb = nb
        self.method = {"names": ["anchor", "prob"], "pages": ["ref_spots", "ref_spots"], "active": 0}
        self.background_images = {"images": [], "names": [], "colours": []}
        self.spots = {"anchor": [], "prob": []}
        self.legend, self.sliders = {}, {}
        # add omp to the method and spots if it is in the notebook
        if nb.has_page("omp"):
            self.method["names"].append("omp")
            self.method["pages"].append("omp")
            self.method["active"] = 2
            self.spots["omp"] = []
        # add genes objects
        self.genes = []
        # populate variables
        self.create_gene_list(gene_marker_file=gene_marker_file, nb_gene_names=nb.call_spots.gene_names)
        self.create_spots_list(nb=nb, downsample_factor=downsample_factor)
        self.load_bg_images(
            nb=nb,
            background_image=background_image,
            background_image_colour=background_image_colour,
            max_intensity_projections=background_image_max_intensity_projection,
            downsample_factor=downsample_factor,
        )

        # create napari viewer
        self.viewer = napari.Viewer()
        self.add_data_to_viewer(spot_size=spot_size / downsample_factor)
        self.update_status_continually()
        self.format_napari_viewer(gene_legend_info=gene_legend_info, nb=nb)
        self.key_call_functions()
        napari.run()

    def format_napari_viewer(self, gene_legend_info: pd.DataFrame, nb: Notebook) -> None:
        """
        Create the napari viewer.
        """
        # turn off layer list and layer controls
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        # add gene legend to viewer
        self.add_legend(gene_legend_info=gene_legend_info)

        # Z-thickness slider
        if nb.basic_info.is_3d:
            # connect a change in z-plane to the z-thickness slider
            self.viewer.dims.events.current_step.connect(self.update_z_thick)
            # connect the z-thickness slider to the update_z_thick function
            self.add_slider(
                name="z_thick",
                value=1,
                slider_range=(0, nb.basic_info.nz),
                slider_mode="single",
                slider_variable="z_thick",
            )

        abs_intensity_thresh = np.percentile(self.spots["anchor"].intensity, 10)
        # set up sliders to adjust background images contrast
        for i, b in enumerate(self.background_images["images"]):
            if b.ndim == 3:
                mid_z = int(b.shape[0] / 2)
                start_contrast = np.percentile(b[mid_z], [50, 100]).astype(int).tolist()
            else:
                start_contrast = np.percentile(b, [50, 100]).astype(int).tolist()
            self.add_slider(
                name=self.background_images["names"][i],
                value=start_contrast,
                slider_range=(b.min(), b.max()),
                slider_mode="range",
                slider_variable="image",
            )

        # Slider to change score_thresh
        self.add_slider(
            name="score_range", value=(0.4, 1), slider_range=(0, 1), slider_mode="range", slider_variable="spot"
        )
        # Slider to change intensity_thresh
        self.add_slider(
            name="intensity_thresh",
            value=(abs_intensity_thresh, np.max(self.spots[self.method["names"][self.method["active"]]].intensity)),
            slider_range=(0, np.nanmax(self.spots[self.method["names"][self.method["active"]]].intensity)),
            slider_mode="range",
            slider_variable="spot",
        )
        # Buttons to change method
        self.method["buttons"] = Method(
            active_button=self.method["names"][self.method["active"]], has_omp=self.method["names"][-1] == "omp"
        )
        for i, m in enumerate(self.method["names"]):
            self.method["buttons"].button[m].clicked.connect(lambda _, x=i: self.method_event_handler(x))
        self.viewer.window.add_dock_widget(self.method["buttons"], area="left", name="Method")

        # label the axes
        if nb.basic_info.is_3d:
            self.viewer.dims.axis_labels = ["z", "y", "x"]
        else:
            self.viewer.dims.axis_labels = ["y", "x"]

    def add_legend(self, gene_legend_info: pd.DataFrame) -> None:
        """
        Add a legend to the napari viewer to show the genes plotted. This will be a figure with the gene names and
        symbols. The self.legend object will contain the figure, axes, xy coordinates and the gene_names of all of the
        genes in the legend and in the notebook.
        """
        # Add legend indicating genes plotted
        legend_gene_names = np.array([gene.name for gene in self.genes if gene.symbol_mpl is not None])
        self.legend["fig"], self.legend["ax"], max_label_length = legend.add_legend(
            gene_legend_info=gene_legend_info,
            genes=legend_gene_names,
        )

        # Initialize positions and gene numbers in the legend
        num_legend_genes = len(legend_gene_names)
        legend_gene_names_cropped = np.array([g[:max_label_length] for g in legend_gene_names])
        self.legend["xy"] = np.zeros((num_legend_genes, 2), dtype=float)
        self.legend["gene_names"] = [None] * num_legend_genes
        # Populate positions and gene numbers for the legend
        for i in range(num_legend_genes):
            self.legend["xy"][i] = np.array(self.legend["ax"].collections[i].get_offsets())
            cropped_legend_gene_name_i = self.legend["ax"].texts[i].get_text()
            gene_i_index = np.where(legend_gene_names_cropped == cropped_legend_gene_name_i)[0][0]
            self.legend["gene_names"][i] = legend_gene_names[gene_i_index]
        # convert the gene names to a numpy array
        self.legend["gene_names"] = np.array(self.legend["gene_names"])

        # Connect the event handler for legend clicks
        self.legend["fig"].mpl_connect("button_press_event", self.legend_event_handler)

        # Add the legend figure to the viewer window
        self.viewer.window.add_dock_widget(self.legend["fig"], area="left", name="Genes")

    def add_slider(
        self,
        name: str,
        value: Union[float, tuple],
        slider_range: tuple,
        slider_mode: str = "single",
        slider_variable: str = "spot",
    ) -> None:
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
        slider.valueChanged.connect(lambda x: self.show_slider_value(string=f"{name}", value=np.array(x)))

        # connect to the appropriate function to update the napari viewer
        if slider_variable == "spot":
            slider.sliderReleased.connect(self.update_genes_and_thresholds)
        elif slider_variable == "image":
            slider.sliderReleased.connect(lambda x=name: self.update_image_contrast(x))
        elif slider_variable == "z_thick":
            slider.sliderReleased.connect(self.update_z_thick)

        # add slider to Viewer object
        self.sliders[name] = slider
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

    def update_image_contrast(self, layer_name: str):
        # Change contrast of background image in napari viewer
        self.viewer.layers[layer_name].contrast_limits = [
            self.sliders[layer_name].value()[0],
            self.sliders[layer_name].value()[1],
        ]

    def update_genes_and_thresholds(self) -> None:
        """
        This method updates the thresholds in the napari viewer to reflect the current state of the Viewer object. It
        will be called when the score or intensity thresholds change.

        The only thresholds this method will update are the score range and intensity threshold.

        The z-thickness slider will be updated in the method `update_z_thick`.
        """
        score_range = self.sliders["score_range"].value()
        intensity_range = self.sliders["intensity_thresh"].value()
        for _, m in enumerate(self.method["names"]):
            # 1. score range
            good_score = (self.spots[m].score >= score_range[0]) & (self.spots[m].score <= score_range[1])
            # 2. intensity threshold
            good_intensity = (self.spots[m].intensity >= intensity_range[0]) & (
                self.spots[m].intensity <= intensity_range[1]
            )
            # 3. gene active
            active_genes = np.array([g.notebook_index for g in self.genes if g.active])
            good_gene = np.isin(self.spots[m].gene, active_genes)
            # create mask
            mask = good_score & good_intensity & good_gene
            self.viewer.layers[m].shown = mask
            self.viewer.layers[m].refresh()

    def update_z_thick(self) -> None:
        """
        This method updates the z-thickness in the napari viewer to reflect the current state of the Viewer object.
        It will be called when the z-thickness changes, or the z-position of the viewer changes.
        """
        range_upper = self.viewer.dims.range[0][1].copy()
        z_thick = self.sliders["z_thick"].value()
        # we will change the z-coordinates of the spots to the current z-plane if they are within the z-thickness
        # of the current z-plane.
        current_z = self.viewer.dims.current_step[0]
        for _, m in enumerate(self.method["names"]):
            z_coords = self.spots[m].location[:, 0].copy()
            in_range = np.abs(z_coords - current_z) <= z_thick / 2
            z_coords[in_range] = current_z
            self.viewer.layers[m].data[:, 0] = z_coords
            self.viewer.layers[m].refresh()

        # napari automatically adjusts the size of the z-step when points are scaled, so we undo that below:
        v_range = list(self.viewer.dims.range)
        v_range[0] = (0, range_upper, 1)
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
                    selected_spot_index = list(selectedData)[0]
                    active_method_name = self.method["names"][self.method["active"]]
                    spot_index = self.spots[active_method_name].notebook_index[selected_spot_index]
                    spot_gene = self.spots[active_method_name].gene[selected_spot_index]
                    score = self.spots[active_method_name].score[selected_spot_index]
                    tile = self.spots[active_method_name].tile[selected_spot_index]
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
                time.sleep(0.1)
                oldSelectedData = copy.deepcopy(selectedData)
                selectedData = copy.deepcopy(pointsLayer.selected_data)
                if oldSelectedData != selectedData:
                    yield selectedData
                yield None

        active_method_name = self.method["names"][self.method["active"]]
        return _watchSelectedData(self.viewer.layers[active_method_name])

    def method_event_handler(self, method_index: int):
        # Set this method as the active method
        self.method["active"] = method_index
        active_layer_name = self.method["names"][method_index]
        self.viewer.layers.selection.active = self.viewer.layers[active_layer_name]

        # loop through methods, turn off all layers except the active one
        # update the buttons to show only the active method as checked
        for i, m in enumerate(self.method["names"]):
            self.viewer.layers[m].visible = i == self.method["active"]
            self.method["buttons"].button[m].setChecked(i == self.method["active"])

    def legend_event_handler(self, event):
        """
        Update the genes plotted in the napari viewer based on the gene that was clicked in the legend.
        When click on a gene in the legend will remove/add that gene to plot.
        When right-click on a gene, it will only show that gene.
        When click on a gene which is the only selected gene, it will return to showing all genes.
        """
        clicked_coordinates = np.array([event.xdata, event.ydata])
        print(f"Clicked coordinates = {clicked_coordinates}")
        closest_gene_coordinates = np.zeros(2)

        # Find the closest gene coordinates in the legend to the clicked position
        for i in range(2):
            closest_grid_point = np.abs(clicked_coordinates[i] - self.legend["xy"][:, i]).argmin()
            closest_gene_coordinates[i] = self.legend["xy"][closest_grid_point, i]

        # Identify the gene that was clicked
        clicked_gene_legend_index = np.where((self.legend["xy"] == closest_gene_coordinates).all(axis=1))[0][0]
        clicked_gene_name = self.legend["gene_names"][clicked_gene_legend_index]
        print(f"Legend says clicked gene = {clicked_gene_name}")
        clicked_gene_notebook_index = np.where([g.name == clicked_gene_name for g in self.genes])[0][0]
        print(f"Notebook says clicked gene = {self.genes[clicked_gene_notebook_index].name}")

        # get the active genes and their indices in the notebook and the legend respectively
        active_gene_notebook_indices = np.array([g.notebook_index for g in self.genes if g.active])
        # get the number of active genes and whether the clicked gene is active
        num_active_genes = len(active_gene_notebook_indices)
        clicked_gene_active = np.isin(clicked_gene_notebook_index, active_gene_notebook_indices)

        # left click will either add or remove the gene from the plot, or if the gene is the only gene shown, will show
        # all genes
        if event.button.name == "LEFT":
            if clicked_gene_active and num_active_genes == 1:
                print("Left click on only active gene")
                # If the gene is the only selected gene, clicking it will show all genes
                for gene in self.genes:
                    gene.active = gene.name in self.legend["gene_names"]
            elif clicked_gene_active and num_active_genes > 1:
                print("Left click on one of many active genes")
                # If single-clicking on an active gene, it will remove that gene
                self.genes[clicked_gene_notebook_index].active = False
            else:
                print("Left click on inactive gene")
                # If single-clicking on an inactive gene, it will add that gene
                self.genes[clicked_gene_notebook_index].active = True
        # right click will either show only that gene (if it is not already the only gene shown) or show all genes
        elif event.button.name == "RIGHT":
            if clicked_gene_active and num_active_genes == 1:
                # If right-clicking on only active gene, it will show all genes
                for gene in self.genes:
                    gene.active = gene.name in self.legend["gene_names"]
            else:
                # if right-clicking in any other case, it will only show that gene
                for gene in self.genes:
                    gene.active = gene.notebook_index == clicked_gene_notebook_index

        # update the lists of active genes and their indices
        active_gene_notebook_indices = np.array([g.notebook_index for g in self.genes if g.active])
        active_gene_names = [self.genes[i].name for i in active_gene_notebook_indices]
        active_gene_legend_indices = np.where(np.isin(self.legend["gene_names"], active_gene_names))[0]

        # Update the legend formatting to reflect the changes
        for i, _ in enumerate(self.legend["gene_names"]):
            alpha_value = 1 if np.isin(i, active_gene_legend_indices) else 0.5
            self.legend["ax"].collections[i].set_alpha(alpha_value)
            self.legend["ax"].texts[i].set_alpha(alpha_value)

        # Redraw the legend figure to apply the changes
        self.legend["fig"].draw()
        self.update_genes_and_thresholds()

    def get_selected_spot_index(self, return_napari_index: bool = False) -> int:
        """
        Get the index of the selected spot. Because the only spots plotted in napari are those that are in the gene
        legend, the index of the spot plotted in napari may differ from the index of the spot in the notebook. This
        function will return the index of the spot in the notebook, unless return_napari_index is True, in which case
        it will return the index of the spot in the napari viewer.
        """
        n_selected = len(self.viewer.layers[self.viewer.layers.selection.active.name].selected_data)
        if n_selected == 1:
            napari_layer_index = list(self.viewer.layers[self.viewer.layers.selection.active.name].selected_data)[0]
            if return_napari_index:
                return napari_layer_index
            spot_index = int(self.spots[self.method["names"][self.method["active"]]].notebook_index[napari_layer_index])
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
         as self.genes. Each element of the list will be a Gene object. So it will contain the name,
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
        legend_gene_names = gene_legend_info["GeneNames"].values
        genes = []

        # create a list of genes with the relevant information. If the gene is not in the gene marker file, it will be
        # added with None values for colour and symbols.
        for i, g in enumerate(nb_gene_names):
            if g in legend_gene_names:
                colour = gene_legend_info[gene_legend_info["GeneNames"] == g][["ColorR", "ColorG", "ColorB"]].values[0]
                symbol_napari = gene_legend_info[gene_legend_info["GeneNames"] == g]["napari_symbol"].values[0]
                symbol_mpl = gene_legend_info[gene_legend_info["GeneNames"] == g]["mpl_symbol"].values[0]
                genes.append(
                    Gene(
                        name=g,
                        notebook_index=i,
                        colour=colour,
                        symbol_mpl=symbol_mpl,
                        symbol_napari=symbol_napari,
                        active=True,
                    )
                )
            else:
                invisible_colour = np.array([0, 0, 0])
                invisible_symbol_mpl = None
                invisible_symbol_napari = None
                genes.append(
                    Gene(
                        name=g,
                        notebook_index=i,
                        colour=invisible_colour,
                        symbol_mpl=invisible_symbol_mpl,
                        symbol_napari=invisible_symbol_napari,
                        active=False,
                    )
                )

        # warn if any genes are not in the gene marker file
        invisible_genes = [g.name for g in genes if not g.active]
        if invisible_genes:
            warnings.warn(f"Genes {invisible_genes} are not in the gene marker file and will not be plotted.")
        self.genes = genes

    def create_spots_list(self, nb: Notebook, downsample_factor: int) -> None:
        """
        Create a list of spots from the notebook to store information about each spot. This will be saved to the object
        as self.spots. Each element of the list will be a Spots object. So it will contain the location, score, tile,
        colour and intensity for each spot.

        Note that self.spots is a list of length n_methods. Each element is a Spots object containing all spots for that
        method.

        This is better than a spots object for each gene as napari does not handle large numbers of layers well.

        Args:
            nb: Notebook containing at least the `call_spots` page.
            downsample_factor: int, factor by which to downsample the images in y and x. Default is 1 which means no
                downsampling.

        """
        # define frequently used variables
        downsample_factor = np.array([1, downsample_factor, downsample_factor])
        tile_origin = nb.stitch.tile_origin
        colour_norm_factor = nb.call_spots.colour_norm_factor

        # initialise relevant information for anchor and prob methods
        tile = [nb.ref_spots.tile, nb.ref_spots.tile]
        local_loc = [nb.ref_spots.local_yxz, nb.ref_spots.local_yxz]
        global_loc = [(local_loc[i] + tile_origin[tile[i]])[:, [2, 0, 1]] for i in range(2)]  # convert to zyx
        # apply downsample factor
        global_loc = [loc // downsample_factor for loc in global_loc]
        colours = [nb.__getattribute__(self.method["pages"][i]).colours.copy() for i in range(2)]
        colours = [colours[i] * colour_norm_factor[tile[i]] for i in range(2)]
        score = [nb.call_spots.dot_product_gene_score, np.max(nb.call_spots.gene_probabilities, axis=1)]
        gene_no = [nb.call_spots.dot_product_gene_no, np.argmax(nb.call_spots.gene_probabilities, axis=1)]
        intensity = [nb.call_spots.intensity, nb.call_spots.intensity]
        indices = [np.arange(len(nb.ref_spots.tile)), np.arange(len(nb.ref_spots.tile))]

        # add omp results to the lists
        if nb.has_page("omp"):
            local_loc_omp, tile_omp = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)
            gene_no_omp = omp_base.get_all_gene_no(nb.basic_info, nb.omp)[0]
            colours_omp = omp_base.get_all_colours(nb.basic_info, nb.omp)[0].copy()
            colours_omp = colours_omp * colour_norm_factor[tile_omp]
            valid_colours = ~np.any(np.isnan(colours_omp), axis=(1, 2))
            score_omp = omp_base.get_all_scores(nb.basic_info, nb.omp)[0]
            intensity_omp = np.median(np.max(colours_omp, axis=-1), axis=-1)
            intensity_omp = np.clip(intensity_omp, 0, np.percentile(intensity_omp[valid_colours], 99))
            indices_omp = np.arange(score_omp.shape[0])

            # convert local_loc_omp to global_loc_omp
            global_loc_omp = local_loc_omp + tile_origin[tile_omp]
            global_loc_omp = global_loc_omp[:, [2, 0, 1]] // downsample_factor

            # append omp results to the lists
            tile.append(tile_omp), local_loc.append(local_loc_omp), global_loc.append(global_loc_omp)
            colours.append(colours_omp), score.append(score_omp), gene_no.append(gene_no_omp)
            intensity.append(intensity_omp), indices.append(indices_omp)

        # add all spots of active genes to the napari viewer. If a gene is not in the legend, the spots assigned to this
        # gene will be disregarded.
        active_genes = np.where([g.active for g in self.genes])[0]
        for i, m in enumerate(self.method["names"]):
            mask = np.isin(gene_no[i], active_genes)
            spot_indices = indices[i][mask]
            self.spots[m] = Spots(
                location=global_loc[i][mask],
                score=score[i][mask],
                tile=tile[i][mask],
                colours=colours[i][mask],
                gene=gene_no[i][mask],
                intensity=intensity[i][mask],
                notebook_index=spot_indices,
            )

    def load_bg_images(
        self,
        background_image: list,
        background_image_colour: list,
        max_intensity_projections: list,
        downsample_factor: int,
        nb: Notebook,
    ) -> None:
        """
        Load background images into the napari viewer and into the Viewer Object.
        Note: napari viewer must be created before calling this function.

        Args:
            background_image: list of file_names or images that will be plotted as the background image.
            background_image_colour: list of names of background colours. Must be same length as background_image
            max_intensity_projections: list of bools. If max_intensity_projections[i] is True, will plot the maximum
                intensity projection of the background image i. If False, will plot the background_image as is.
            downsample_factor: int, factor by which to downsample the images in y and x. Default is 1 which means no
                downsampling.
            nb: Notebook.

        """
        # if background image is None, quit the function
        if background_image is None:
            print("No background image given, so not plotting any background images.")
            return
        # make things lists if not already
        if not isinstance(background_image, list):
            background_image = [background_image]
        if not isinstance(background_image_colour, list):
            background_image_colour = [background_image_colour]
        if not isinstance(max_intensity_projections, list):
            max_intensity_projections = [max_intensity_projections]
        # if only 1 colour has been given, repeat it for all images
        if len(background_image_colour) == 1 and len(background_image) > 1:
            print("Only 1 colour given, repeating for all images")
            background_image_colour = background_image_colour * len(background_image)
        # if only 1 max_intensity_projection has been given, repeat it for all images
        if len(max_intensity_projections) == 1 and len(background_image) > 1:
            print("Only 1 max_intensity_projection preference given, repeating for all images")
            max_intensity_projections = max_intensity_projections * len(background_image)
        # now assert that all lists are the same length
        assert (
            len(background_image) == len(background_image_colour) == len(max_intensity_projections)
        ), "background_image, background_image_colour and max_intensity_projections must all be the same length."
        # assert that downsample_factor is an integer
        assert isinstance(downsample_factor, int), "downsample_factor must be an integer."

        # load images
        background_image_dir = []
        # users were allowed to specify a list of strings or images. If a string, assume it is a file name and try to
        # load it. Save file names in background_image_dir and images in background_image.
        for i, b in enumerate(background_image):
            # Check if b is a string, if so, assume it is a file name and try to load it. Then load the image into
            # background_image. If it is not a string, assume it is an image and load it directly into background_image.
            if not isinstance(b, str):
                background_image_dir.append(None)
                continue
            # If we have got this far, b is a string. Check if it is a keyword for a standard image.
            if b.lower() in ("dapi", "anchor"):
                file_name = b.lower() + "_image"
            else:
                file_name = b
            background_image_dir.append(file_name)
            # Now try to load the image
            if file_name is not None and os.path.isfile(file_name):
                if file_name.endswith(".npz"):
                    # Assume image is first array if .npz file. Now replace the string with the actual image.
                    # Note:
                    background_image[i] = np.load(file_name, mmap_mode="r")["arr_0"]
                    background_image[i] = background_image[i][:, ::downsample_factor, ::downsample_factor]
                elif file_name.endswith(".npy"):
                    background_image[i] = np.load(file_name, mmap_mode="r")
                    background_image[i] = background_image[i][:, ::downsample_factor, ::downsample_factor]
                elif file_name.endswith(".tif"):
                    with tifffile.TiffFile(file_name) as tif:
                        background_image[i] = tif.asarray()[:, ::downsample_factor, ::downsample_factor]
            else:
                background_image[i] = nb.stitch.__getattribute__(file_name)[:][
                    :, ::downsample_factor, ::downsample_factor
                ]
            # If the user specified MIP[i] = True, plot the maximum intensity projection of the image.
            if background_image[i] is not None and max_intensity_projections[i]:
                background_image[i] = background_image[i].max(axis=0)
            # Check if background image is constant. If so, don't plot it.
            if background_image[i] is not None and np.allclose(
                [background_image[i].max()], [background_image[i].min()]
            ):
                warnings.warn(
                    f"Background image with file name =\n\t{file_name}" + "\ncontains constant values, so not plotting"
                )
                background_image[i] = None

        # remove none entries from bg_images
        good = [i for i, b in enumerate(background_image) if b is not None]
        self.background_images["images"] = [background_image[i] for i in good]
        self.background_images["names"] = [os.path.basename(background_image_dir[i]).split(".")[0] for i in good]
        self.background_images["colours"] = [background_image_colour[i] for i in good]

    def add_data_to_viewer(self, spot_size: float = 10) -> None:
        """
        Add background images and spots to the napari viewer (they already exist in the Viewer object).

        The user can specify the size of the spots to be plotted. This will be the size of the spots in the napari
        viewer.

        There will be n_bg image layers, followed by n_methods (either 2 or 3) spots layers. This is because a single
        layer for each gene of each method is too slow to render in napari.
        """
        # Loop through all background images and add them to the viewer.
        for i, b in enumerate(self.background_images["images"]):
            self.viewer.add_image(
                b,
                blending="additive",
                colormap=self.background_images["colours"][i],
                name=self.background_images["names"][i],
                contrast_limits=[np.percentile(b, 50), np.percentile(b, 100)],
            )
            self.viewer.layers[i].contrast_limits_range = [b.min(), b.max()]

        # add a single layer of spots to viewer for each method
        active_genes = np.array([g.notebook_index for g in self.genes if g.active])
        gene_colours = np.array([g.colour for g in self.genes])
        gene_symbols = np.array([g.symbol_napari for g in self.genes])
        for i, m in enumerate(self.method["names"]):
            point_loc = self.spots[m].location.copy()
            # load the colours and symbols for the spots of this method. These are both arrays of length n_spots.
            gene_no = self.spots[m].gene
            face_colour = gene_colours[gene_no]
            face_symbol = gene_symbols[gene_no]
            # face_colour is an array of shape (n_spots, 4) where the last dimension is the rgba values.
            # Set final row to all 1s
            face_colour = np.hstack((face_colour, np.ones((face_colour.shape[0], 1))))
            mask_gene = np.isin(gene_no, active_genes)
            mask_score = (self.spots[m].score >= 0.4) & (self.spots[m].score <= 1)
            mask_intensity = self.spots[m].intensity >= 0.25
            mask = mask_gene & mask_score & mask_intensity
            self.viewer.add_points(
                data=point_loc,
                size=spot_size,
                face_color=face_colour,
                symbol=face_symbol,
                name=m,
                out_of_slice_display=False,
                visible=(i == self.method["active"]),
                shown=mask,
            )
        self.viewer.layers.selection.active = self.viewer.layers[self.method["names"][self.method["active"]]]

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
                    self.background_image_name[i]
                ].visible

        @self.viewer.bind_key(KeyBinds.view_bleed_matrix)
        def call_to_view_bm(viewer):
            ViewBleedMatrix(self.nb)

        @self.viewer.bind_key(KeyBinds.view_background_norm)
        def call_to_view_bg_norm(viewer):
            ViewScalingAndBGRemoval(self.nb)

        @self.viewer.bind_key(KeyBinds.view_bled_codes)
        def call_to_view_bled(viewer):
            view_bled_codes(self.nb)

        @self.viewer.bind_key(KeyBinds.view_all_gene_scores)
        def call_to_view_all_hists(viewer):
            ViewAllGeneHistograms(self.nb)

        @self.viewer.bind_key(KeyBinds.view_gene_efficiency)
        def call_to_view_gene_efficiency(viewer):
            self.open_plot = GeneEfficiencyViewer(
                self.nb,
                mode=self.method["names"][self.method["active"]],
                score_threshold=self.sliders["score_range"].value()[0],
            )

        @self.viewer.bind_key(KeyBinds.view_histogram_scores)
        def call_to_view_omp_score(viewer):
            HistogramScore(self.nb)

        @self.viewer.bind_key(KeyBinds.view_scaled_k_means)
        def call_to_view_omp_score(viewer):
            call_spots_plot.view_scaled_k_means(self.nb)

        @self.viewer.bind_key(KeyBinds.view_colour_and_codes)
        def call_to_view_codes(viewer):
            notebook_index = self.get_selected_spot_index()
            napari_index = self.get_selected_spot_index(return_napari_index=True)
            spot_tile = self.spots[self.method["names"][self.method["active"]]].tile[napari_index]
            if notebook_index is not None:
                view_codes(self.nb, notebook_index, spot_tile, self.method["names"][self.method["active"]])

        @self.viewer.bind_key(KeyBinds.view_spot_intensities)
        def call_to_view_spot(viewer):
            spot_index = self.get_selected_spot_index()
            napari_index = self.get_selected_spot_index(return_napari_index=True)
            spot_tile = int(self.spots[self.method["names"][self.method["active"]]].tile[napari_index])
            if spot_index is not None:
                view_spot(self.nb, spot_index, spot_tile, self.method["names"][self.method["active"]])

        # @self.viewer.bind_key(KeyBinds.view_spot_colours_and_weights)
        # def call_to_view_omp_score(viewer):
        #     spot_index = self.get_selected_spot_index()
        #     if spot_index is not None:
        #         view_score(self.nb, spot_index, self.method["names"][self.method["active"]])

        @self.viewer.bind_key(KeyBinds.view_omp_coef_image)
        def call_to_view_omp(viewer):
            spot_index = self.get_selected_spot_index()
            if spot_index is not None:
                self.open_plot = ViewOMPImage(self.nb, spot_index, self.method["names"][self.method["active"]])

        @self.viewer.bind_key(KeyBinds.view_omp_pixel_colours)
        def call_to_view_omp_colours(viewer):
            spot_index = self.get_selected_spot_index()
            if spot_index is not None:
                self.open_plot = ViewOMPPixelColours(self.nb, spot_index, self.method["names"][self.method["active"]])


class Method(QMainWindow):
    def __init__(self, active_button: str = "anchor", has_omp: bool = True):
        """
        Create a window with buttons to change between anchor, prob and omp spots. Will have the attributes:
        button_prob, button_anchor, button_omp (if has_omp is True).
        Args:
            active_button: (str) name of the button that should be active initially. Must be one of "anchor", "prob" or
                "omp" (if has_omp is True).
            has_omp: (bool) whether the notebook has an OMP page.
        """
        assert active_button in ["anchor", "prob", "omp"]
        assert has_omp or active_button != "omp"

        super().__init__()
        self.button = {"anchor": QPushButton("anchor", self), "prob": QPushButton("prob", self)}

        # Set up the buttons
        self.button["anchor"].setCheckable(True)
        self.button["anchor"].setGeometry(50, 2, 50, 28)  # left, top, width, height
        self.button["prob"].setCheckable(True)
        self.button["prob"].setGeometry(105, 2, 50, 28)  # left, top, width, height

        if has_omp:
            self.button["omp"] = QPushButton("omp", self)
            self.button["omp"].setCheckable(True)
            self.button["omp"].setGeometry(160, 2, 50, 28)

        self.button[active_button].setChecked(True)


class Spots:
    """
    Class to hold different spot information. In the Viewer class we will have a list of lists of Spots objects, one
    for each gene within each method.
    """

    def __init__(
        self,
        location: np.ndarray,
        colours: np.ndarray,
        score: np.ndarray,
        tile: np.ndarray,
        intensity: np.ndarray,
        gene: np.ndarray,
        notebook_index: np.ndarray,
    ):
        """
        Create object for spots of a single gene within a single method.
        Args:
            location: (np.ndarray) of shape (n_spots, 3) with the zyx location of each spot. (int16)
            colours: (np.ndarray) of shape (n_spots, n_rounds, n_channels) of the raw colour of each spot. (uint16)
            score: (np.ndarray) of shape (n_spots,) with the score of each spot. (float32)
            tile: (np.ndarray) of shape (n_spots,) with the tile of each spot. (int16)
            intensity: (np.ndarray) of shape (n_spots,) with the intensity of each spot. (float32)
            gene: (np.ndarray) of shape (n_spots,) with the gene number of each spot. (int16)
            notebook_index: (np.ndarray) of shape (n_spots,) with the index of each spot in the notebook. (int16)
        """
        assert (
            len(location)
            == len(colours)
            == len(score)
            == len(tile)
            == len(intensity)
            == len(gene)
            == len(notebook_index)
        ), "All arrays must be the same length."
        self.location = location
        self.colours = colours
        self.score = score
        self.tile = tile
        self.intensity = intensity
        self.gene = gene
        self.notebook_index = notebook_index


class Gene:
    """
    Class to hold gene information. In the Viewer class we will have a list of Gene objects, one for each gene.
    This will store the gene name, index, colour and symbol.
    """

    def __init__(
        self,
        name: str,
        notebook_index: int,
        colour: Union[np.ndarray, None],
        symbol_mpl: Union[str, None],
        symbol_napari: Union[str, None],
        active: bool = True,
    ):
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
            active: (bool) whether the gene is active or not in the viewer.
        """
        self.name = name
        self.notebook_index = notebook_index
        self.colour = colour
        self.symbol_mpl = symbol_mpl
        self.symbol_napari = symbol_napari
        self.active = active
