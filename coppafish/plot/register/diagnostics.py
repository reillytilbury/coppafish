from PyQt5.QtWidgets import QLabel, QLineEdit, QMainWindow, QPushButton, QSlider
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import napari
import nd2
import numpy as np
import skimage
import torch

from qtpy.QtCore import Qt
from scipy.ndimage import affine_transform
from scipy.spatial import KDTree
from superqt import QRangeSlider
from tqdm import tqdm
from coppafish.find_spots import spot_yxz
from coppafish.register import preprocessing
from coppafish.setup import Notebook
from coppafish.spot_colours import apply_affine, apply_flow


class RegistrationViewer:
    def __init__(self, nb: Notebook, t: int = None):
        """
        Viewer for the registration of an experiment.
        - Shows the registered images for the selected tile.
        - Allows to adjust contrast limits for the images.
        - Allows to switch on/off the imaging and anchor images.
        - Allows to switch between tiles.

        Developer Note: No need to add too many variables to this class. Best to keep them in the napari viewer which
        can be accessed by self.viewer.
        Args:
            nb: Notebook object (should contain register and register_debug pages)
            t: tile (if None, the first tile is selected)
        """
        self.nb = nb
        if t is None:
            t = nb.basic_info.use_tiles[0]
        self.t = t
        self.viewer = napari.Viewer()
        self.add_images()
        self.format_viewer()
        napari.run()

    def add_images(self):
        """
        Load images for the selected tile and add them to the viewer.
        """
        # load round images
        round_im, channel_im = {}, {}
        for r in list(self.nb.basic_info.use_rounds):
            round_im[f"r{r}"] = self.nb.register.round_images[self.t, r]
        # repeat anchor image 3 times along new 0 axis
        im_anchor = self.nb.register.anchor_images[self.t, 0]
        round_im["anchor"] = np.repeat(im_anchor[None], 3, axis=0)
        # load channel images
        for c in list(self.nb.basic_info.use_channels):
            channel_im[f"c{c}"] = self.nb.register.channel_images[self.t, c]
        # repeat anchor image 3 times along new 0 axis
        im_anchor = self.nb.register.anchor_images[self.t, 1]
        channel_im["anchor"] = np.repeat(im_anchor[None], 3, axis=0)

        # clear previous images
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        yx_size = round_im["r0"].shape[1]
        unit_step = yx_size * 1.1
        # add round images
        for i, r in enumerate(self.nb.basic_info.use_rounds):
            offset = tuple([0, 0, i * unit_step, 0])
            self.viewer.add_image(
                round_im[f"r{r}"], name=f"r{r}", blending="additive", colormap="green", translate=offset
            )
            self.viewer.add_image(
                round_im["anchor"], name="anchor_dapi", blending="additive", colormap="red", translate=offset
            )
        # add channel images
        for i, c in enumerate(self.nb.basic_info.use_channels):
            offset = tuple([0, unit_step, i * unit_step, 0])
            self.viewer.add_image(
                channel_im[f"c{c}"],
                name=f"c{c}",
                blending="additive",
                colormap="green",
                translate=offset,
                contrast_limits=(30, 255),
            )
            self.viewer.add_image(
                channel_im["anchor"],
                name="anchor_seq",
                blending="additive",
                colormap="red",
                translate=offset,
                contrast_limits=(10, 180),
            )
        # label axes
        self.viewer.dims.axis_labels = ["method", "y", "x", "z"]
        # set default order for axes as (method, z, y, x)
        self.viewer.dims.order = (0, 3, 1, 2)
        # Add points to attach text
        n_methods, n_z = 3, round_im["r0"].shape[-1]
        n_rounds = len(round_im)
        mid_x = unit_step * (n_rounds - 1) // 2
        points = [[i, -unit_step // 4, mid_x, z] for i in range(n_methods) for z in range(n_z)]
        method_names = ["unregistered", "optical flow", "optical flow + ICP"]
        text = {
            "string": [f"tile: {self.t}, method: {method_names[i]}" for i in range(n_methods) for _ in range(n_z)],
            "color": "white",
            "size": 10,
        }
        self.viewer.add_points(points, size=0, text=text, name="text")

    def get_layer_ind(self, layer: str):
        """
        Get the index of the layers in the viewer.
        Args:
            layer: either 'round', 'round_anchor', 'channel', or 'channel_anchor' or 'imaging' or 'anchor'

        Returns:
            layer_ind: list of indices of the layers in the viewer
        """
        if layer == "round":
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[0] == "r"]
        elif layer == "round_anchor":
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name == "anchor_dapi"]
        elif layer == "channel":
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[0] == "c"]
        elif layer == "channel_anchor":
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name == "anchor_seq"]
        elif layer == "imaging":
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[0] in ["r", "c"]]
        elif layer == "anchor":
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[:6] == "anchor"]
        else:
            raise ValueError(f"Layer {layer} is not recognized.")
        return layer_ind

    def format_viewer(self):
        """
        Format the viewer.
        """
        # Make layer list invisible to remove clutter
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        # add sliders to adjust contrast limits
        self.add_contrast_lim_sliders()
        # add buttons to switch on/off the layers
        self.add_switch_button()
        # add buttons to switch between tiles
        self.add_tile_buttons()
        # add buttons to view optical flow results
        self.add_optical_flow_buttons()
        # add buttons to view overlay
        self.add_overlay_buttons()
        # add buttons to view ICP correction and ICP iterations
        self.add_icp_buttons()
        # add buttons to view camera correction
        self.add_fluorescent_bead_buttons()

    # Functions to add buttons and sliders to the viewer
    def add_contrast_lim_sliders(self):
        # add contrast limits sliders
        contrast_limit_sliders = [QRangeSlider(Qt.Horizontal) for _ in range(2)]
        labels = ["imaging", "anchor"]
        layer_ind = [self.get_layer_ind("imaging"), self.get_layer_ind("anchor")]
        # add these to the viewer and connect them to the appropriate functions
        for i, slider in enumerate(contrast_limit_sliders):
            self.viewer.window.add_dock_widget(slider, area="left", name=labels[i])
            slider.setRange(0, 255)
            slider.setValue((0, 255))
            slider.valueChanged.connect(lambda value, j=i: self.update_contrast_limits(layer_ind[j], value))

    def add_switch_button(self):
        # add buttons to switch on/off the layers
        switch_button = ButtonCreator(["anchor", "imaging"], [(50, 2), (140, 2)])
        self.viewer.window.add_dock_widget(switch_button, area="left", name="switch")
        switch_button.buttons[0].clicked.connect(lambda: self.toggle_layers(self.get_layer_ind("anchor")))
        switch_button.buttons[1].clicked.connect(lambda: self.toggle_layers(self.get_layer_ind("imaging")))

    def add_tile_buttons(self):
        # add buttons to select tile to view
        n_tiles_use = len(self.nb.basic_info.use_tiles)
        button_loc = generate_button_positions(n_buttons=n_tiles_use, n_cols=4)
        button_name = [f"t{t}" for t in self.nb.basic_info.use_tiles]
        button = ButtonCreator(button_name, button_loc, size=(50, 28))
        self.viewer.window.add_dock_widget(button, area="left", name="tiles")
        for i, b in enumerate(button.buttons):
            b.clicked.connect(lambda _, t=self.nb.basic_info.use_tiles[i]: self.switch_tile(t))

    def add_optical_flow_buttons(self):
        # add buttons to select round to view (for optical flow overlay and optical flow vector field)
        use_rounds = self.nb.basic_info.use_rounds
        n_rounds_use = len(use_rounds)
        button_loc = generate_button_positions(n_buttons=n_rounds_use, n_cols=4)
        button_name = [f"r{r}" for r in use_rounds]
        button = ButtonCreator(button_name, button_loc, size=(50, 28))
        for i, b in enumerate(button.buttons):
            b.clicked.connect(lambda _, r=use_rounds[i]: view_optical_flow(self.nb, self.t, r))
        self.viewer.window.add_dock_widget(button, area="left", name="optical flow viewer")

    def add_overlay_buttons(self):
        # add button to view background scale
        button_loc = generate_button_positions(n_buttons=5, n_cols=2, x_offset=50, x_spacing=100)
        button_name = ["r1", "c1", "r2", "c2"]
        text_buttons = TextButtonCreator(button_name, button_loc, size=(50, 28))
        text_buttons.button.clicked.connect(
            lambda _: view_overlay(
                self.nb,
                self.t,
                rc=[
                    (int(text_buttons.text_box[0].text()), int(text_buttons.text_box[1].text())),
                    (int(text_buttons.text_box[2].text()), int(text_buttons.text_box[3].text())),
                ]
            )
        )
        self.viewer.window.add_dock_widget(text_buttons, area="left", name="overlay")

    def add_icp_buttons(self):
        # add buttons to view ICP correction and ICP iterations
        use_tiles, use_rounds, use_channels = (
            self.nb.basic_info.use_tiles,
            self.nb.basic_info.use_rounds,
            self.nb.basic_info.use_channels,
        )
        n_tiles_use, n_rounds_use, n_channels_use = len(use_tiles), len(use_rounds), len(use_channels)
        # get all button locations
        button_loc_1 = generate_button_positions(n_buttons=n_tiles_use, n_cols=4)
        button_loc_2 = generate_button_positions(
            n_buttons=n_tiles_use, n_cols=4, y_offset=35 + np.max(button_loc_1[:, 1])
        )
        button_loc_3 = generate_button_positions(
            n_buttons=n_rounds_use, n_cols=4, y_offset=35 + np.max(button_loc_2[:, 1])
        )
        button_loc_4 = generate_button_positions(
            n_buttons=n_channels_use, n_cols=4, y_offset=35 + np.max(button_loc_3[:, 1])
        )
        button_loc = np.concatenate([button_loc_1, button_loc_2, button_loc_3, button_loc_4])
        # name all buttons (tile, tile iter, round, channel)
        button_name = (
            [f"t{t} diff" for t in use_tiles]
            + [f"t{t} iter" for t in use_tiles]
            + [f"r{r}" for r in use_rounds]
            + [f"c{c}" for c in use_channels]
        )
        button = ButtonCreator(button_name, button_loc, size=(50, 28))
        # loop through and link buttons to the appropriate functions
        for i, b in enumerate(button.buttons):
            if i < n_tiles_use:
                b.clicked.connect(lambda _, t=use_tiles[i]: view_icp_correction(self.nb, t))
            elif n_tiles_use <= i < 2 * n_tiles_use:
                i -= n_tiles_use
                b.clicked.connect(lambda _, t=use_tiles[i]: view_icp_iters(self.nb, t))
            elif 2 * n_tiles_use <= i < 2 * n_tiles_use + n_rounds_use:
                i -= 2 * n_tiles_use
                b.clicked.connect(lambda _, r=use_rounds[i]: ICPPointCloudViewer(self.nb, self.t, r=r))
            else:
                i -= 2 * n_tiles_use + n_rounds_use
                b.clicked.connect(lambda _, c=use_channels[i]: ICPPointCloudViewer(self.nb, self.t, c=c))
        self.viewer.window.add_dock_widget(button, area="left", name="icp")

    def add_fluorescent_bead_buttons(self):
        # add buttons to view camera correction
        button = ButtonCreator(["fluorescent beads"], [(60, 5)], size=(150, 28))
        button.buttons[0].clicked.connect(lambda: view_camera_correction(self.nb))
        if self.nb.file_names.fluorescent_bead_path is not None:
            self.viewer.window.add_dock_widget(button, area="left", name="camera correction")

    # Functions to interact with the viewer
    def update_contrast_limits(self, layer_ind: list, contrast_limits: tuple):
        """
        Update contrast limits for the selected layers.
        Args:
            layer_ind: list of indices of the layers in the viewer
            contrast_limits: tuple of contrast limits
        """
        for i in layer_ind:
            self.viewer.layers[i].contrast_limits = [contrast_limits[0], contrast_limits[1]]

    def toggle_layers(self, layer_ind: list):
        """
        Toggle layers on/off.
        Args:
            layer_ind: list of indices of the layers in the viewer
        """
        for i in layer_ind:
            self.viewer.layers[i].visible = not self.viewer.layers[i].visible

    def switch_tile(self, t: int):
        """
        Switch to a different tile.
        Args:
            t: tile number
        """
        self.t = t
        self.add_images()


class ButtonCreator(QMainWindow):
    def __init__(self, names: list, position: np.ndarray, size: tuple = (75, 28)):
        """
        Create buttons.
        Args:
            names: list of n_buttons names (str)
            position: array of n_buttons positions (x, y) for the buttons
            size: (width, height) of the buttons
        """
        super().__init__()
        assert len(names) == len(position), "Number of names and positions should be the same."
        self.buttons = []
        for i, name in enumerate(names):
            self.buttons.append(QPushButton(name, self))
            self.buttons[-1].setCheckable(True)
            self.buttons[-1].setGeometry(position[i][0], position[i][1], size[0], size[1])


class TextButtonCreator(QMainWindow):
    def __init__(self, names: list, position: np.ndarray, size: tuple = (75, 28)):
        """
        Create buttons with text.
        Args:
            names: list of n_buttons names (str)
            position: array of n_buttons positions (x, y) for the buttons
            size: (width, height) of the buttons
        """
        super().__init__()
        assert len(names) == len(position) - 1, "Number of names should be one less than the number of positions."
        self.text_box = []
        for i, name in enumerate(names):
            label = QLabel(self)
            label.setText(name)
            label.setGeometry(position[i][0] - 30, position[i][1], size[0], size[1])
            self.text_box.append(QLineEdit(name, self))
            self.text_box[-1].setGeometry(position[i][0], position[i][1], size[0], size[1])
            self.text_box[-1].setText("0")
        self.button = QPushButton("view", self)
        self.button.setGeometry(position[-1][0], position[-1][1], size[0], size[1])


class ICPPointCloudViewer:
    def __init__(self, nb: Notebook, t: int, r: int = None, c: int = None):
        """
        Visualize the point cloud registration results for the selected tile and round.
        Args:
            nb: Notebook object (must have register and register_debug pages and find_spots page
            t: tile to view
            r: round to view
            c: channel to view
        NOTE! If r == None, then we are in channel mode. If c == None, then we are in round mode. Both are not allowed
        to be None.
        """
        assert r is not None or c is not None, "Either r or c should be provided."
        assert r is None or c is None, "Only one of r or c should be provided."
        self.nb = nb
        self.t, self.r, self.c = t, r, c
        self.z_thick = 1
        self.anchor_round, self.anchor_channel = nb.basic_info.anchor_round, nb.basic_info.anchor_channel
        # initialize the points (these will be: base_untransformed, base_1, base_2, target)
        # base untransformed and target will always be the same, but base 1 and base 2 be different depending on
        # the mode we are in.
        # base_1: mode r: flow_r(base_untransformed), mode c: affine_r(flow_r(base_untransformed))
        # base_2: mode r: icp_correction_r(base_1), mode c: icp_correction_c(base_1)
        self.points, self.matching_points = [], []
        self.score = None
        self.z_thick_slider = None
        # fill in the points, matching points, and create the density map
        self.get_spot_data()
        self.create_density_map()
        # create viewer
        self.viewer = napari.Viewer()
        self.add_toggle_base_button()
        self.add_z_thickness_slider()
        self.view_data()
        # connect a z-thickness update to changes in the z-slider
        if self.z_thick_slider is not None:
            self.viewer.dims.events.current_step.connect(self.update_z_thick)
        # if r not None then we want to show all z-planes
        if r is not None:
            self.viewer.dims.order = (1, 0, 2)
        napari.run()

    def get_spot_data(self, dist_thresh: int = 5, down_sample_yx: int = 10):
        # remove points that exist already
        self.points = []
        # Step 1: Get the points
        # get anchor points
        base_raw = spot_yxz(
            self.nb.find_spots.spot_yxz, self.t, self.anchor_round, self.anchor_channel, self.nb.find_spots.spot_no
        )

        # get base points after applying the flow and the affine round correction
        if self.r is None:
            # in channel mode, take middle round (r=3) and appy the flow followed by the affine correction
            r = 3
            affine_round_correction = self.nb.register_debug.round_correction[self.t, 3]
        else:
            # in round mode, apply the flow only and set the round to the selected round
            r = self.r
            affine_round_correction = np.eye(4, 3)
        flow = self.nb.register.flow[self.t, r]
        # apply the flow and affine correction
        base_flow_plus_round_affine = apply_flow(yxz=base_raw, flow=flow)
        base_flow_plus_round_affine = apply_affine(yxz=torch.asarray(base_flow_plus_round_affine, dtype=torch.float32),
                                                   affine=torch.asarray(affine_round_correction,
                                                                        dtype=torch.float32)).numpy()
        # remove out of bounds points
        in_bounds = (np.all(base_flow_plus_round_affine[:, :2] >= 0, axis=1)
                     & np.all(base_flow_plus_round_affine[:, :2] < self.nb.basic_info.tile_sz, axis=1))
        base_raw = base_raw[in_bounds]
        base_flow_plus_round_affine = base_flow_plus_round_affine[in_bounds]

        # get base points after applying flow and icp correction
        if self.r is None:
            icp_correction = self.nb.register_debug.channel_correction[self.t, self.c]
        else:
            icp_correction = self.nb.register_debug.round_correction[self.t, self.r]
        # apply the icp correction
        base_flow_plus_icp_affine = apply_affine(yxz=torch.asarray(base_flow_plus_round_affine, dtype=torch.float32),
                                                 affine=torch.asarray(icp_correction, dtype=torch.float32)).numpy()
        # remove out of bounds points
        in_bounds = (np.all(base_flow_plus_icp_affine[:, :2] >= 0, axis=1)
                     & np.all(base_flow_plus_icp_affine[:, :2] < self.nb.basic_info.tile_sz, axis=1))
        base_flow_plus_icp_affine = base_flow_plus_icp_affine[in_bounds]
        base_flow_plus_round_affine = base_flow_plus_round_affine[in_bounds]
        base_raw = base_raw[in_bounds]

        # get target points
        if self.r is None:
            r = 3
            c = self.c
        if self.c is None:
            r = self.r
            c = self.anchor_channel
        target = spot_yxz(self.nb.find_spots.spot_yxz, self.t, r, c, self.nb.find_spots.spot_no)
        self.points.append(base_raw)
        self.points.append(base_flow_plus_round_affine)
        self.points.append(base_flow_plus_icp_affine)
        self.points.append(target)

        # convert to zyx
        for i in range(4):
            self.points[i] = self.points[i][:, [2, 0, 1]].astype(np.float32)

        # Step 2: Find overlapping points
        # get the nearest neighbour in base_2 for each point in target
        tree = KDTree(self.points[-2])
        dist, nearest_ind = tree.query(self.points[-1])
        matching_round_spots = np.where(dist < dist_thresh)[0]
        matching_anchor_spots = nearest_ind[matching_round_spots]

        # Step 3: Downsample the points (this will make the density map faster and doesn't affect our precision)
        # down-sample the points in xy
        down_sample_factor = np.array([1, down_sample_yx, down_sample_yx])
        self.points = [p / down_sample_factor for p in self.points]

        # convert matching spots from indices to points
        matching_anchor_spots_old = self.points[0][matching_anchor_spots]
        matching_anchor_spots = self.points[-2][matching_anchor_spots]
        matching_round_spots = self.points[-1][matching_round_spots]
        self.matching_points = [matching_anchor_spots_old, matching_anchor_spots, matching_round_spots]

    def create_density_map(self):
        # get the mid point of the matching points
        mid_point = np.round((self.matching_points[0] + self.matching_points[1]) / 2).astype(int)
        # Create indicator function of the mid points
        nz, ny, nx = (
            self.nb.basic_info.nz + 2,
            self.nb.basic_info.tile_sz // 10 + 1,
            self.nb.basic_info.tile_sz // 10 + 1,
        )
        score = np.zeros((nz, ny, nx), dtype=np.float32)
        score[mid_point[:, 0], mid_point[:, 1], mid_point[:, 2]] = 1
        self.score = skimage.filters.gaussian(score, sigma=2, truncate=3)

    def add_z_thickness_slider(self):
        # add slider to adjust z thickness
        if self.r is None:
            self.z_thick_slider = QSlider(Qt.Orientation.Horizontal)
            self.z_thick_slider.setRange(1, self.nb.basic_info.nz)
            self.z_thick_slider.setValue(self.z_thick)
            self.z_thick_slider.sliderReleased.connect(self.update_z_thick)
            # add the slider to the viewer
            self.viewer.window.add_dock_widget(self.z_thick_slider, area="left", name="z thickness")

    def add_toggle_base_button(self):
        # add button to toggle between base and base_1
        button = QPushButton("Toggle Base", self.viewer.window.qt_viewer)
        button.setGeometry(20, 5, 60, 28)
        button.clicked.connect(self.toggle_base)
        # add button to the viewer
        self.viewer.window.add_dock_widget(button, area="left", name="toggle base")

    def update_z_thick(self) -> None:
        """
        This method updates the z-thickness in the napari viewer to reflect the current state of the Viewer object.
        It will be called when the z-thickness changes, or the z-position of the viewer changes.
        """
        range_upper = self.viewer.dims.range[0][1].copy()
        # we will change the z-coordinates of the spots to the current z-plane if they are within the z-thickness
        # of the current z-plane.
        current_z = self.viewer.dims.current_step[0]
        z_thick = self.z_thick_slider.value()
        # layers 0, 1, 2 are points
        # 3, 4 are lines
        # 5 is the score

        # adjust the z-coordinates of the points layers
        for i in range(3):
            z_coords = self.viewer.layers[i].data[:, 0].copy()
            in_range = np.abs(z_coords - current_z) <= z_thick / 2
            z_coords[in_range] = current_z
            self.viewer.layers[i].data[:, 0] = z_coords
            self.viewer.layers[i].refresh()

        # adjust the z-coords of the lines layers
        for i in range(3, 5):
            line_coords = np.array(self.viewer.layers[i].data.copy())
            # get the z-coords and modify them
            line_coords_z_lower, line_coords_z_upper = line_coords[:, 0, 0].copy(), line_coords[:, 1, 0].copy()
            in_range_lower = np.abs(line_coords_z_lower - current_z) <= z_thick / 2
            in_range_upper = np.abs(line_coords_z_upper - current_z) <= z_thick / 2
            in_range = in_range_lower & in_range_upper
            line_coords_z_lower[in_range] = current_z
            line_coords_z_upper[in_range] = current_z
            # update the line coords array with the new z-coords
            line_coords[:, 0, 0] = line_coords_z_lower
            line_coords[:, 1, 0] = line_coords_z_upper
            self.viewer.layers[i].data = list(line_coords)
            self.viewer.layers[i].refresh()

        # napari automatically adjusts the size of the z-step when points are scaled, so we undo that below:
        v_range = list(self.viewer.dims.range)
        v_range[0] = (0, range_upper, 1)
        self.viewer.dims.range = tuple(v_range)

    def view_data(self):
        # turn off default napari widgets
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        # define the colours and symbols
        name = ["base_unregistered", "base_registered", "target"]
        colours = ["white", "white", "red"]
        symbols = ["o", "o", "x"]
        visible = [False, True, True]
        # add the points
        for i in range(3):
            self.viewer.add_points(
                self.points[i + 1],
                size=0.5,
                face_color=colours[i],
                symbol=symbols[i],
                visible=visible[i],
                opacity=0.6,
                name=name[i],
                out_of_slice_display=False,
            )
        # add line between the matching points
        line_locs_old = []
        line_locs = []
        for i in range(len(self.matching_points[0])):
            line_locs_old.append([self.matching_points[0][i], self.matching_points[2][i]])
            line_locs.append([self.matching_points[1][i], self.matching_points[2][i]])
        mse_old = np.mean(np.linalg.norm(self.matching_points[0] - self.matching_points[2], axis=1))
        mse_new = np.mean(np.linalg.norm(self.matching_points[1] - self.matching_points[2], axis=1))
        self.viewer.add_shapes(
            line_locs_old,
            shape_type="line",
            edge_color="cyan",
            edge_width=0.25,
            name=f"mse_old: {mse_old:.2f}",
            visible=False,
        )
        self.viewer.add_shapes(
            line_locs, shape_type="line", edge_color="blue", edge_width=0.25, name=f"mse_new: {mse_new:.2f}"
        )

        # add the score image
        self.viewer.add_image(self.score, name="score", colormap="bop orange", blending="additive", opacity=0.7)
        self.viewer.dims.axis_labels = ["z", "y", "x"]

    def toggle_base(self):
        #  toggle between (layer 0 on, layer 1 off) and (layer 0 off, layer 1 on)
        self.viewer.layers[0].visible = not self.viewer.layers[0].visible
        self.viewer.layers[1].visible = not self.viewer.layers[1].visible
        # also toggle the lines
        self.viewer.layers[-3].visible = not self.viewer.layers[-3].visible
        self.viewer.layers[-2].visible = not self.viewer.layers[-2].visible


def generate_button_positions(
    n_buttons: int, n_cols: int, x_offset: int = 5, y_offset: int = 5, x_spacing: int = 60, y_spacing: int = 35
):
    """
    Generate positions for the buttons.
    Args:
        n_buttons: number of buttons
        n_cols: number of columns
        x_offset: x offset for the first button
        y_offset: y offset for the first button
        x_spacing: spacing between buttons in x
        y_spacing: spacing between buttons in y

    Returns:
        button_positions: np.ndarray of shape (n_buttons, 2) with x and y positions for each button
    """
    x = x_offset + x_spacing * (np.arange(n_buttons) % n_cols)
    y = y_offset + y_spacing * (np.arange(n_buttons) // n_cols)
    button_positions = np.array([(x[i], y[i]) for i in range(n_buttons)])
    return button_positions


def view_optical_flow(nb: Notebook, t: int, r: int):
    """
    Visualize the optical flow results using napari.
    Args:
        nb: Notebook (containing register and register_debug pages)
        t: tile number
        r: round number
    """
    # Load the data
    base = nb.filter.images[t, 7, 0].astype(np.float32)
    target = nb.filter.images[t, r, 0].astype(np.float32)
    ny, nx, nz = base.shape
    coord_order = ["y", "x", "z"]
    coords = np.array(np.meshgrid(range(ny), range(nx), range(nz), indexing="ij"), dtype=np.float32)
    print("Base and Target images loaded.")
    # load the warps
    names = ["raw", "smooth"]
    flows = [nb.register.flow_raw[t, r], nb.register.flow[t, r]]
    # warp the base image using the flows
    base_warped = [skimage.transform.warp(base, -f.astype(np.float32) + coords, order=0, preserve_range=True)
                   for f in flows]
    print("Base image warped.")
    del coords
    # load the correlation
    corr = nb.register.correlation[t, r]
    print("Correlation loaded.")

    # create viewer
    viewer = napari.Viewer()
    # add overlays
    viewer.add_image(target, name="target", colormap="green", blending="additive")
    viewer.add_image(base, name="base", colormap="red", blending="additive")
    for i in range(len(flows)):
        translation = [0, 1.1 * nx * (i + 1), 0]
        viewer.add_image(target, name="target", colormap="green", blending="additive", translate=translation)
        viewer.add_image(
            base_warped[i],
            name=names[i],
            colormap="red",
            blending="additive",
            translate=translation
        )
    # add flows as images
    for i, j in np.ndindex(len(flows), len(coord_order)):
        translation = [1.1 * ny * (j + 1), 1.1 * nx * (i + 1), 0]
        viewer.add_image(
            flows[i][j], name=names[i] + " : " + coord_order[j], translate=translation, contrast_limits=[-10, 10]
        )
    # add correlation
    for i in range(1):
        translation = [1.1 * ny * (len(coord_order) + 1), 1.1 * nx * (i + 1), 0]
        viewer.add_image(corr, name="correlation: " + names[i], colormap="cyan", translate=translation)

    # label axes
    viewer.dims.axis_labels = ["y", "x", "z"]
    # set default order for axes as (method, z, y, x)
    viewer.dims.order = (2, 0, 1)
    # run napari
    napari.run()

    # plot the average shifts for each z-plane
    mean_flow = [np.mean(f, axis=(1, 2)) for f in flows]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(2):
        ax[i].plot(mean_flow[i].T)
        ax[i].set_title(names[i])
        ax[i].set_xlabel("Z")
        ax[i].set_ylabel("Average shift")
        ax[i].legend(["Y", "X", "Z"])

    plt.suptitle("Average shifts for tile " + str(t) + " and round " + str(r))
    plt.show()


def view_icp_correction(nb: Notebook, t: int):
    """
    Visualize the ICP correction results in 6 subplots: (scale corrections in y, x, z), (shift corrections in y, x, z)
    Args:
        nb: Notebook object (must have register and register debug pages)
        t: tile to view

    """
    use_rounds = nb.basic_info.use_rounds
    use_channels = nb.basic_info.use_channels
    transform = nb.register.icp_correction[t][np.ix_(use_rounds, use_channels)].copy()
    scale, shift = np.zeros((len(use_rounds), len(use_channels), 3)), np.zeros((len(use_rounds), len(use_channels), 3))
    # populate scales and shifts
    for r, c in np.ndindex(len(use_rounds), len(use_channels)):
        transform_rc = transform[r, c]
        scale[r, c] = transform_rc[:3, :3].diagonal()
        shift[r, c] = transform_rc[3]

    # create plots
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    coord_label = ["Y", "X", "Z"]
    plot_label = ["Scale", "Shift"]
    image = [scale, shift]

    for i, j in np.ndindex(2, 3):
        # plot the image
        ax[i, j].imshow(image[i][:, :, j].T)
        ax[i, j].set_xticks(np.arange(len(use_rounds)))
        ax[i, j].set_xticklabels(use_rounds)
        ax[i, j].set_yticks(np.arange(len(use_channels)))
        ax[i, j].set_yticklabels(use_channels)
        ax[i, j].set_title(plot_label[i] + " in " + coord_label[j])
        # for each subplot, assign a colour bar
        divider = make_axes_locatable(ax[i, j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(ax[i, j].get_images()[0], cax=cax)

        if j == 0:
            ax[i, j].set_ylabel("Channel")
        if i == 1:
            ax[i, j].set_xlabel("Round")
    plt.suptitle("Shifts and scales for tile " + str(t))
    plt.show()


def view_icp_iters(nb: Notebook, t: int):
    """
    Visualize the ICP iterations for the selected tile.
    Will show 4 rows and several columns of plots. Each row will show the following:
    - 1: MSE vs iteration for round corrections
    - 2: MSE vs iteration for channel corrections
    - 3: Frac_match vs iteration for round corrections
    - 4: Frac_match vs iteration for channel corrections
    Args:
        nb: Notebook object (must have register and register debug pages)
        t: tile to view
    """
    # get the data
    use_rounds = nb.basic_info.use_rounds
    use_channels = nb.basic_info.use_channels
    n_rounds, n_channels = len(use_rounds), len(use_channels)
    anchor_channel = nb.basic_info.anchor_channel
    spot_no = nb.find_spots.spot_no[t]
    mse = [nb.register_debug.mse_round[t, use_rounds], nb.register_debug.mse_channel[t, use_channels]]
    n_matches = [nb.register_debug.n_matches_round[t, use_rounds], nb.register_debug.n_matches_channel[t, use_channels]]
    frac_matches = n_matches.copy()
    for i, r in enumerate(use_rounds):
        # calculate the fraction of matches
        frac_matches[0][i] = n_matches[0][i] / spot_no[r, anchor_channel]
    for i, c in enumerate(use_channels):
        # calculate the fraction of matches
        total_spots = np.sum(spot_no[use_rounds, c])
        frac_matches[1][i] = n_matches[1][i] / total_spots
    # create plots
    n_cols = max(n_rounds, n_channels)
    fig, ax = plt.subplots(4, n_cols, figsize=(4 * n_cols, 10))

    data = mse + frac_matches
    labels = ["MSE Round", "MSE Channel", "Frac Match Round", "Frac Match Channel"]
    indices = [use_rounds, use_channels, use_rounds, use_channels]
    y_max = [np.max(data[0]), np.max(data[1]), 1, 1]
    for i in range(4):
        n_cols_current = len(data[i])
        for j in range(n_cols_current):
            ax[i, j].plot(data[i][j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlim([0, len(data[i][j]) // 2])
            ax[i, j].set_ylim([0, y_max[i]])
            ax[i, j].set_title(indices[i][j])
        for j in range(n_cols_current, n_cols):
            ax[i, j].axis("off")
        ax[i, 0].set_ylabel(labels[i])
        ax[i, 0].set_yticks([0, y_max[i]])
        ax[i, 0].set_yticklabels([0, round(y_max[i], 2)])
    plt.suptitle("ICP iterations for tile " + str(t))
    plt.show()


def view_camera_correction(nb: Notebook):
    """
    Plots the camera correction for each camera against the anchor camera
    Args:
        nb: Notebook (must have register page and a path to fluorescent bead images)
    """
    # One transform for each camera
    viewer = napari.Viewer()
    fluorescent_bead_path = nb.file_names.fluorescent_bead_path
    # open the fluorescent bead images as nd2 files
    with nd2.ND2File(fluorescent_bead_path) as fbim:
        fluorescent_beads = fbim.asarray()

    if len(fluorescent_beads.shape) == 4:
        mid_z = fluorescent_beads.shape[0] // 2
        fluorescent_beads = fluorescent_beads[mid_z, :, :, :]
    # if fluorescent bead images are for all channels, just take one from each camera
    cam_channels = [0, 9, 18, 23]
    if len(fluorescent_beads) == 28:
        fluorescent_beads = fluorescent_beads[cam_channels]

    # set the beads to const intensity
    for i in range(4):
        threshold = skimage.filters.threshold_isodata(fluorescent_beads[i])
        bead_pixels = fluorescent_beads[i] > threshold
        fluorescent_beads[i][bead_pixels] = threshold

    # obtain the initial transform for each channel
    transform = np.repeat(np.eye(2, 3)[None, :, :], 4, axis=0)
    for i, c in enumerate(cam_channels):
        transform[i][:2, :2] = nb.register_debug.channel_transform_initial[c][:2, :2].T
        transform[i][:2, -1] = nb.register_debug.channel_transform_initial[c][-1, :2]

    # get the spots from the circle detection
    bead_point_clouds = []
    bead_radii = nb.get_config()["register"]["bead_radii"]
    for i in range(4):
        edges = skimage.feature.canny(fluorescent_beads[i], sigma=3, low_threshold=10, high_threshold=50)
        hough_res = skimage.transform.hough_circle(edges, bead_radii)
        accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
            hough_res, bead_radii, min_xdistance=10, min_ydistance=10
        )
        cy, cx = cy.astype(int), cx.astype(int)
        values = fluorescent_beads[i][cy, cx]
        cy_rand, cx_rand = (
            np.random.randint(0, fluorescent_beads[i].shape[0] - 1, 100),
            np.random.randint(0, fluorescent_beads[i].shape[1] - 1, 100),
        )
        noise = np.mean(fluorescent_beads[i][cy_rand, cx_rand])
        keep = values > noise
        cy, cx = cy[keep], cx[keep]
        bead_point_clouds.append(np.vstack((cy, cx)).T)

    # Apply the transform to the fluorescent bead images
    fluorescent_beads_transformed = np.zeros(fluorescent_beads.shape)
    for c in range(3):
        fluorescent_beads_transformed[c] = affine_transform(fluorescent_beads[c], transform[c], order=3)
    # The last channel is the anchor channel (no transform)
    fluorescent_beads_transformed[3] = np.copy(fluorescent_beads[3])

    # Transform the bead point clouds to the anchor frame of reference
    bead_point_clouds_transformed = []
    for i in range(3):
        points = np.hstack((bead_point_clouds[c], np.ones((bead_point_clouds[c].shape[0], 1))))
        affine = np.linalg.inv(np.vstack((transform[c], [0, 0, 1])))
        points_transformed = points @ affine.T
        bead_point_clouds_transformed.append(points_transformed)

    # Add the images to napari
    colours = ["red", "green", "blue"]
    for c in range(1, 3):
        # add unregistered points and images
        viewer.add_image(
            fluorescent_beads[c],
            name=f"Camera {cam_channels[c]} image",
            colormap=colours[c],
            blending="additive",
            visible=False,
        )
        viewer.add_points(
            bead_point_clouds[c],
            name=f"Camera {cam_channels[c]} points",
            face_color=colours[c],
            symbol="o",
            size=5,
            blending="additive",
            visible=False,
        )
        # add registered points and images
        viewer.add_image(
            fluorescent_beads_transformed[c],
            name=f"Camera {cam_channels[c]} transformed image",
            colormap=colours[c],
            blending="additive",
            visible=True,
        )
        viewer.add_points(
            bead_point_clouds_transformed[c],
            name=f"Camera {cam_channels[c]} transformed points",
            face_color=colours[c],
            symbol="x",
            size=5,
            blending="additive",
            visible=True,
        )
    # Add the anchor channel image and points
    viewer.add_image(
        fluorescent_beads[-1],
        name=f"Camera {cam_channels[-1]} image",
        colormap=colours[-1],
        blending="additive",
        visible=False,
    )
    viewer.add_points(
        bead_point_clouds[-1],
        name=f"Camera {cam_channels[-1]} points",
        face_color="white",
        symbol="o",
        size=5,
        blending="additive",
        visible=False,
    )

    napari.run()


def view_overlay(nb: Notebook, t: int = None, rc: list = None):
    """
    Visualize the overlay of two images, both in the anchor frame of reference.
    Args:
        nb: Notebook object (must have register and register_debug pages)
        t: common tile
        rc: list of length n_images where rc[i] = (r, c) for the i-th image
    """
    assert len(rc) > 0, "At least one round and channel should be provided."
    n_im = len(rc)
    coords = np.array(np.meshgrid(range(nb.basic_info.tile_sz), range(nb.basic_info.tile_sz),
                                  range(nb.basic_info.nz), indexing="ij"), dtype=np.float32)
    im = np.zeros((n_im, nb.basic_info.tile_sz, nb.basic_info.tile_sz, nb.basic_info.nz), dtype=np.float32)

    icp_correction = nb.register.icp_correction
    icp_correction[:, :, nb.basic_info.dapi_channel] = icp_correction[:, :, nb.basic_info.anchor_channel]
    # load, affine correct, and flow correct the images
    for i, rc_pair in tqdm(enumerate(rc), total=len(rc), desc="Loading images"):
        # LOAD IMAGE
        r, c = rc_pair
        # if the anchor round, no need tp apply registration
        if r == nb.basic_info.anchor_round:
            im[i] = nb.filter.images[t, r, c].astype(np.float32)
        else:
            # don't use get spot colours as that loads all rounds
            im[i] = nb.filter.images[t, r, c].astype(np.float32)
            # apply the affine to the image (first have to adjust the shift for the new origin)
            im[i] = affine_transform(im[i], icp_correction[t, r, c].T, order=0)
            # apply the flow to the image
            flow = nb.register.flow[t, r]
            im[i] = skimage.transform.warp(im[i], coords + flow, order=0, preserve_range=True, cval=0, mode="constant")
    # create viewer
    viewer = napari.Viewer()
    colours = ["red", "green"]
    for i, rc_pair in enumerate(rc):
        r, c = rc_pair
        viewer.add_image(im[i], name=f"t{t}_r{r}_c{c}", colormap=colours[i], blending="additive")
    viewer.dims.axis_labels = ["y", "x", "z"]
    viewer.dims.order = (2, 0, 1)
    napari.run()