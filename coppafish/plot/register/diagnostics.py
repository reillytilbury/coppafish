from PyQt5.QtWidgets import QPushButton, QMainWindow, QSlider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from coppafish.spot_colors import apply_transform
from coppafish.utils.tiles_io import load_image
from scipy.ndimage import affine_transform
from coppafish.register import preprocessing
from coppafish.find_spots import spot_yxz
from coppafish.setup import Notebook
from skimage.transform import warp
from scipy.spatial import KDTree
from superqt import QRangeSlider
from qtpy.QtCore import Qt
import matplotlib.pyplot as plt
import numpy as np
import skimage
import napari
import nd2
import os


class RegistrationViewer():
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
        self.reg_data_dir = os.path.join(self.nb.file_names.output_dir, 'reg_images', f't{self.t}')
        self.viewer = napari.Viewer()
        self.add_images()
        self.format_viewer()
        napari.run()

    def add_images(self):
        """
        Load images for the selected tile and add them to the viewer.
        """
        # get directory for the selected tile
        self.reg_data_dir = os.path.join(self.nb.file_names.output_dir, 'reg_images', f't{self.t}')
        # load round images
        round_im, channel_im = {}, {}
        for r in self.nb.basic_info.use_rounds:
            round_im[f'r{r}'] = np.load(os.path.join(self.reg_data_dir, 'round', f'r{r}.npy'))
        # repeat anchor image 3 times along new 0 axis
        im_anchor = np.load(os.path.join(self.reg_data_dir, 'round', 'anchor.npy'))
        round_im['anchor'] = np.repeat(im_anchor[None], 3, axis=0)
        # load channel images
        for c in self.nb.basic_info.use_channels:
            channel_im[f'c{c}'] = np.load(os.path.join(self.reg_data_dir, 'channel', f'c{c}.npy'))
        # repeat anchor image 3 times along new 0 axis
        im_anchor = np.load(os.path.join(self.reg_data_dir, 'channel', 'anchor.npy'))
        channel_im['anchor'] = np.repeat(im_anchor[None], 3, axis=0)

        # clear previous images
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        yx_size = round_im['r0'].shape[1]
        unit_step = yx_size * 1.1
        # add round images
        for i, r in enumerate(self.nb.basic_info.use_rounds):
            offset = tuple([0, 0, i * unit_step, 0])
            self.viewer.add_image(round_im[f'r{r}'], name=f'r{r}', blending='additive', colormap='green',
                                  translate=offset)
            self.viewer.add_image(round_im['anchor'], name='anchor_dapi', blending='additive', colormap='red',
                                  translate=offset)
        # add channel images
        for i, c in enumerate(self.nb.basic_info.use_channels):
            offset = tuple([0, unit_step, i * unit_step, 0])
            self.viewer.add_image(channel_im[f'c{c}'], name=f'c{c}', blending='additive', colormap='green',
                                  translate=offset,
                                  contrast_limits=(30, 255))
            self.viewer.add_image(channel_im['anchor'], name='anchor_seq', blending='additive', colormap='red',
                                  translate=offset, contrast_limits=(10, 180))
        # label axes
        self.viewer.dims.axis_labels = ['method', 'y', 'x', 'z']
        # set default order for axes as (method, z, y, x)
        self.viewer.dims.order = (0, 3, 1, 2)
        # Add points to attach text
        n_methods, n_z = 3, round_im['r0'].shape[-1]
        n_rounds = len(round_im)
        mid_x = unit_step * (n_rounds - 1) // 2
        points = [[i, -unit_step // 4, mid_x, z] for i in range(n_methods) for z in range(n_z)]
        method_names = ['unregistered', 'optical flow', 'optical flow + ICP']
        text = {'string': [f'tile: {self.t}, method: {method_names[i]}' for i in range(n_methods) for _ in range(n_z)],
                'color': 'white', 'size': 10}
        self.viewer.add_points(points, size=0, text=text, name='text')

    def get_layer_ind(self, layer: str):
        """
        Get the index of the layers in the viewer.
        Args:
            layer: either 'round', 'round_anchor', 'channel', or 'channel_anchor' or 'imaging' or 'anchor'

        Returns:
            layer_ind: list of indices of the layers in the viewer
        """
        if layer == 'round':
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[0] == 'r']
        elif layer == 'round_anchor':
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name == 'anchor_dapi']
        elif layer == 'channel':
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[0] == 'c']
        elif layer == 'channel_anchor':
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name == 'anchor_seq']
        elif layer == 'imaging':
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[0] in ['r', 'c']]
        elif layer == 'anchor':
            layer_ind = [self.viewer.layers.index(l) for l in self.viewer.layers if l.name[:6] == 'anchor']
        else:
            raise ValueError(f'Layer {layer} is not recognized.')
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
        # add buttons to view ICP correction and ICP iterations
        self.add_icp_buttons()
        # add buttons to view camera correction
        self.add_fluorescent_bead_buttons()

    def add_contrast_lim_sliders(self):
        # add contrast limits sliders
        contrast_limit_sliders = [QRangeSlider(Qt.Horizontal) for _ in range(2)]
        labels = ['imaging', 'anchor']
        layer_ind = [self.get_layer_ind('imaging'), self.get_layer_ind('anchor')]
        # add these to the viewer and connect them to the appropriate functions
        for i, slider in enumerate(contrast_limit_sliders):
            self.viewer.window.add_dock_widget(slider, area="left", name=labels[i])
            slider.setRange(0, 255)
            slider.setValue((0, 255))
            slider.valueChanged.connect(lambda value, j=i: self.update_contrast_limits(layer_ind[j], value))

    def add_switch_button(self):
        # add buttons to switch on/off the layers
        switch_button = ButtonCreator(['anchor', 'imaging'], [(50, 2), (140, 2)])
        self.viewer.window.add_dock_widget(switch_button, area="left", name='switch')
        switch_button.buttons[0].clicked.connect(lambda: self.toggle_layers(self.get_layer_ind('anchor')))
        switch_button.buttons[1].clicked.connect(lambda: self.toggle_layers(self.get_layer_ind('imaging')))

    def add_tile_buttons(self):
        # add buttons to select tile to view
        n_tiles_use = len(self.nb.basic_info.use_tiles)
        button_loc = generate_button_positions(n_buttons=n_tiles_use, n_cols=4)
        button_name = [f't{t}' for t in self.nb.basic_info.use_tiles]
        button = ButtonCreator(button_name, button_loc, size=(50, 28))
        self.viewer.window.add_dock_widget(button, area="left", name='tiles')
        for i, b in enumerate(button.buttons):
            b.clicked.connect(lambda _, t=self.nb.basic_info.use_tiles[i]: self.switch_tile(t))

    def add_optical_flow_buttons(self):
        # add buttons to select round to view (for optical flow overlay and optical flow vector field)
        use_rounds = (self.nb.basic_info.use_rounds +
                      [self.nb.basic_info.pre_seq_round] * self.nb.basic_info.use_preseq)
        n_rounds_use = len(use_rounds)
        button_loc = generate_button_positions(n_buttons=n_rounds_use, n_cols=4)
        button_name = [f'r{r}' for r in use_rounds]
        button = ButtonCreator(button_name, button_loc, size=(50, 28))
        for i, b in enumerate(button.buttons):
            b.clicked.connect(lambda _, r=use_rounds[i]: view_optical_flow(self.nb, self.t, r))
        self.viewer.window.add_dock_widget(button, area="left", name='optical flow viewer')

    def add_icp_buttons(self):
        # add buttons to view ICP correction and ICP iterations
        use_tiles, use_rounds, use_channels = self.nb.basic_info.use_tiles, self.nb.basic_info.use_rounds, \
                                                self.nb.basic_info.use_channels
        n_tiles_use, n_rounds_use, n_channels_use = len(use_tiles), len(use_rounds), len(use_channels)
        # get all button locations
        button_loc_1 = generate_button_positions(n_buttons=n_tiles_use, n_cols=4)
        button_loc_2 = generate_button_positions(n_buttons=n_tiles_use, n_cols=4,
                                                 y_offset=35 + np.max(button_loc_1[:, 1]))
        button_loc_3 = generate_button_positions(n_buttons=n_rounds_use, n_cols=4,
                                                 y_offset=35 + np.max(button_loc_2[:, 1]))
        button_loc_4 = generate_button_positions(n_buttons=n_channels_use, n_cols=4,
                                                 y_offset=35 + np.max(button_loc_3[:, 1]))
        button_loc = np.concatenate([button_loc_1, button_loc_2, button_loc_3, button_loc_4])
        # name all buttons (tile, tile iter, round, channel)
        button_name = [f't{t}' for t in use_tiles] + [f't{t} iter' for t in use_tiles] + \
                      [f'r{r}' for r in use_rounds] + [f'c{c}' for c in use_channels]
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
        self.viewer.window.add_dock_widget(button, area="left", name='icp')

    def add_fluorescent_bead_buttons(self):
        # add buttons to view camera correction
        button = ButtonCreator(['fluorescent beads'], [(60, 5)], size=(150, 28))
        button.buttons[0].clicked.connect(lambda: view_camera_correction(self.nb))
        if self.nb.file_names.fluorescent_bead_path is not None:
            self.viewer.window.add_dock_widget(button, area="left", name='camera correction')

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
        assert len(names) == len(position), 'Number of names and positions should be the same.'
        self.buttons = []
        for i, name in enumerate(names):
            self.buttons.append(QPushButton(name, self))
            self.buttons[-1].setCheckable(True)
            self.buttons[-1].setGeometry(position[i][0], position[i][1], size[0], size[1])


class ICPPointCloudViewer():
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
        assert r is not None or c is not None, 'Either r or c should be provided.'
        assert r is None or c is None, 'Only one of r or c should be provided.'
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
        # fill in the points, matching points, and create the density map
        self.get_spot_data()
        self.create_density_map()
        # create viewer
        self.viewer = napari.Viewer()
        self.add_toggle_base_button()
        self.add_z_thickness_slider()
        self.view_data()
        napari.run()

    def get_spot_data(self, dist_thresh: int = 5, down_sample_yx: int = 10):
        # remove points that exist already
        self.points = []
        # Step 1: Get the points
        # get anchor points
        base = spot_yxz(self.nb.find_spots.spot_yxz, self.t, self.anchor_round, self.anchor_channel,
                        self.nb.find_spots.spot_no)

        # get base 1 points
        if self.r is None:
            # in channel mode, take middle round (r=3) and appy the flow followed by the affine correction
            r = 3
            affine_round_correction = self.nb.register_debug.round_correction[self.t, 3]
        else:
            # in round mode, apply the flow only and set the round to the selected round
            r = self.r
            affine_round_correction = np.eye(4, 3)
        flow = np.load(os.path.join(self.nb.register.flow_dir, "smooth", f"t{self.t}_r{r}.npy"), mmap_mode='r')
        base_1, in_bounds = apply_transform(yxz=base, flow=flow, icp_correction=affine_round_correction,
                                            tile_sz=self.nb.basic_info.tile_sz)
        base_1 = base_1[in_bounds]
        base = base[in_bounds]

        # get base 2 points
        if self.r is None:
            icp_correction = self.nb.register_debug.channel_correction[self.t, self.c]
        else:
            icp_correction = self.nb.register_debug.round_correction[self.t, self.r]

        base_2, in_bounds = apply_transform(yxz=base_1, flow=None, icp_correction=icp_correction,
                                            tile_sz=self.nb.basic_info.tile_sz)
        base_2 = base_2[in_bounds]
        base = base[in_bounds]

        # get target points
        if self.r is None:
            r = 3
            c = self.c
        if self.c is None:
            r = self.r
            c = self.anchor_channel
        target = spot_yxz(self.nb.find_spots.spot_yxz, self.t, r, c, self.nb.find_spots.spot_no)
        self.points.append(base)
        self.points.append(base_1)
        self.points.append(base_2)
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
        nz, ny, nx = (self.nb.basic_info.nz + 2,
                      self.nb.basic_info.tile_sz // 10 + 1,
                      self.nb.basic_info.tile_sz // 10 + 1)
        score = np.zeros((nz, ny, nx), dtype=np.float32)
        score[mid_point[:, 0], mid_point[:, 1], mid_point[:, 2]] = 1
        self.score = skimage.filters.gaussian(score, sigma=2, truncate=3)

    def add_z_thickness_slider(self):
        # add slider to adjust z thickness
        z_size_slider = QSlider(Qt.Orientation.Horizontal)
        z_size_slider.setRange(1, self.nb.basic_info.nz)
        z_size_slider.setValue(self.z_thick)
        z_size_slider.valueChanged.connect(lambda x: self.adjust_z_thickness(x))
        # add the slider to the viewer
        self.viewer.window.add_dock_widget(z_size_slider, area="left", name='z thickness')

    def add_toggle_base_button(self):
        # add button to toggle between base and base_1
        button = QPushButton('Toggle Base', self.viewer.window.qt_viewer)
        button.setGeometry(20, 5, 60, 28)
        button.clicked.connect(self.toggle_base)
        # add button to the viewer
        self.viewer.window.add_dock_widget(button, area="left", name='toggle base')

    def adjust_z_thickness(self, val: int):
        self.z_thick = val
        for layer in self.viewer.layers:
            layer.size = [val, 0.5, 0.5]

    def view_data(self):
        # turn off default napari widgets
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)
        # define the colours and symbols
        name = ['base_unregistered', 'base_registered', 'target']
        colours = ['white', 'white', 'red']
        symbols = ['o', 'o', 'x']
        visible = [False, True, True]
        # add the points
        for i in range(3):
            self.viewer.add_points(self.points[i + 1], size=[self.z_thick, 0.5, 0.5], face_color=colours[i],
                                   symbol=symbols[i], visible=visible[i], opacity=0.6, name=name[i],
                                   out_of_slice_display=True)
        # add line between the matching points
        line_locs_old = []
        line_locs = []
        for i in range(len(self.matching_points[0])):
            line_locs_old.append([self.matching_points[0][i], self.matching_points[2][i]])
            line_locs.append([self.matching_points[1][i], self.matching_points[2][i]])
        mse_old = np.mean(np.linalg.norm(self.matching_points[0] - self.matching_points[2], axis=1))
        mse_new = np.mean(np.linalg.norm(self.matching_points[1] - self.matching_points[2], axis=1))
        self.viewer.add_shapes(line_locs_old, shape_type='line', edge_color='cyan', edge_width=0.25,
                               name=f'mse_old: {mse_old:.2f}', visible=False)
        self.viewer.add_shapes(line_locs, shape_type='line', edge_color='blue', edge_width=0.25,
                               name=f'mse_new: {mse_new:.2f}')

        # add the score image
        self.viewer.add_image(self.score, name='score', colormap='bop orange', blending='additive', opacity=0.7)
        self.viewer.dims.axis_labels = ['z', 'y', 'x']

    def toggle_base(self):
        #  toggle between (layer 0 on, layer 1 off) and (layer 0 off, layer 1 on)
        self.viewer.layers[0].visible = not self.viewer.layers[0].visible
        self.viewer.layers[1].visible = not self.viewer.layers[1].visible
        # also toggle the lines
        self.viewer.layers[-3].visible = not self.viewer.layers[-3].visible
        self.viewer.layers[-2].visible = not self.viewer.layers[-2].visible


def generate_button_positions(n_buttons: int, n_cols: int, x_offset: int = 5, y_offset: int = 5,
                              x_spacing: int = 60, y_spacing: int = 35):
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
    base = load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=7, c=0)
    target = load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r, c=0)
    ny, nx, nz = base.shape
    coord_order = ['y', 'x', 'z']
    coords = np.array(np.meshgrid(range(ny), range(nx), range(nz), indexing='ij'))
    print('Base and Target images loaded.')
    # load the warps
    output_dir = nb.file_names.output_dir + '/flow'
    name = ['raw', 'smooth']
    flow = [np.load(os.path.join(output_dir, f, f't{t}_r{r}.npy')).astype(np.float32) for f in name]
    # warp the base image using the flows
    base_warped = [skimage.transform.warp(base, f + coords, order=0) for f in flow]
    print('Base image warped.')
    del coords
    # load the correlation
    corr = np.load(os.path.join(output_dir, 'corr', f't{t}_r{r}.npy'))
    print('Correlation loaded.')
    mask = corr > np.percentile(corr, 97.5)

    # create viewer
    viewer = napari.Viewer()
    # add overlays
    viewer.add_image(target, name='target', colormap='green', blending='additive')
    viewer.add_image(base, name='base', colormap='red', blending='additive')
    for i in range(len(flow)):
        translation = [0, 1.1 * nx * (i + 1), 0]
        viewer.add_image(target, name='target', colormap='green', blending='additive', translate=translation)
        viewer.add_image(base_warped[i], name=name[i], colormap='red', blending='additive',
                         translate=translation, contrast_limits=(0, 5_000))
    # add flows as images
    for i, j in np.ndindex(len(flow), len(coord_order)):
        translation = [1.1 * ny * (j + 1), 1.1 * nx * (i + 1), 0]
        viewer.add_image(flow[i][j], name=name[i] + ' : ' + coord_order[j], translate=translation,
                         contrast_limits=[-10, 10])
        if i == 0:
            viewer.add_image(mask, name='mask', colormap='red', translate=translation, opacity=0.2, blending='additive')
    # add correlation
    for i in range(1):
        translation = [1.1 * ny * (len(coord_order) + 1), 1.1 * nx * (i + 1), 0]
        viewer.add_image(corr, name='correlation: ' + name[i], colormap='cyan', translate=translation)

    # label axes
    viewer.dims.axis_labels = ['y', 'x', 'z']
    # set default order for axes as (method, z, y, x)
    viewer.dims.order = (2, 0, 1)
    # run napari
    napari.run()


# def view_flow_vector_field(nb: Notebook, t: int, r: int):
#     """
#     Visualize the optical flow results using napari.
#     Args:
#         nb: Notebook (containing register and register_debug pages)
#         t: tile number
#         r: round number
#     """
#     # load the flow
#     output_dir = nb.file_names.output_dir + '/flow'
#     flow = np.load(os.path.join(output_dir, 'smooth', f't{t}_r{r}.npy'), mmap_mode='r')[:, ::100, ::100, ::5]
#     flow = flow.astype(np.float32)
#     ny, nx, nz = flow.shape[1:]
#     start_points = np.array(np.meshgrid(range(ny), range(nx), range(nz), indexing='ij'))
#     flow = np.moveaxis(flow, 0, -1)
#     flow = flow.reshape(ny * nx * nz, 3)
#     start_points = np.moveaxis(start_points, 0, -1)
#     start_points = start_points.reshape(ny * nx * nz, 3)
#     vectors = np.array([start_points, flow])
#     vectors = np.moveaxis(vectors, 0, 1)
#     print('Flow loaded.')
#     # create colourmap for the flow
#     cmap = napari.utils.Colormap([[0, 0, 1, 0], [1, 0, 0, 1]], name='blue_red', interpolation='linear')
#     flow_max = np.max(np.linalg.norm(flow, axis=-1))
#
#     # create viewer
#     viewer = napari.Viewer()
#     viewer.add_vectors(vectors, name='flow', edge_width=0.4, length=0.6,
#                        properties={'magnitude': np.linalg.norm(flow, axis=-1)},
#                        edge_colormap=cmap, edge_contrast_limits=[0, flow_max])
#     viewer.dims.axis_labels = ['y', 'x', 'z']
#     viewer.dims.order = (2, 0, 1)
#     napari.run()


def view_icp_correction(nb: Notebook, t: int):
    """
    Visualize the ICP correction results in 6 subplots: (scale corrections in y, x, z), (shift corrections in y, x, z)
    Args:
        nb: Notebook object (must have register and register debug pages)
        t: tile to view

    """
    use_rounds = nb.basic_info.use_rounds
    use_channels = nb.basic_info.use_channels
    transform = nb.register.icp_correction[t][np.ix_(use_rounds, use_channels)]
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
        if nb.basic_info.use_preseq:
            total_spots = np.sum(spot_no[use_rounds[:-1], c])
        else:
            total_spots = np.sum(spot_no[use_rounds, c])
        frac_matches[1][i] = n_matches[1][i] / total_spots
    # create plots
    n_cols = max(n_rounds, n_channels)
    fig, ax = plt.subplots(4, n_cols, figsize=(4 * n_cols, 10))

    data = mse + frac_matches
    labels = ['MSE Round', 'MSE Channel', 'Frac Match Round', 'Frac Match Channel']
    indices = [use_rounds, use_channels, use_rounds, use_channels]
    y_max = [np.max(data[0]), np.max(data[1]), 1, 1]
    for i in range(4):
        n_cols_current = len(data[i])
        for j in range(n_cols_current):
            ax[i, j].plot(data[i][j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_xlim([0, len(data[i][j])//2])
            ax[i, j].set_ylim([0, y_max[i]])
            ax[i, j].set_title(indices[i][j])
        for j in range(n_cols_current, n_cols):
            ax[i, j].axis('off')
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
    transform = nb.register_debug.channel_transform_initial[cam_channels]
    linear_transform = transform[:, :2, :2]
    shift = transform[:, 3, :2]
    # put these together to form the full transform
    transform = np.zeros((4, 3, 3))
    transform[:, :2, :2] = linear_transform
    transform[:, :2, 2] = shift
    transform[:, 2, 2] = 1

    # Apply the transform to the fluorescent bead images
    fluorescent_beads_transformed = np.zeros(fluorescent_beads.shape)
    for c in range(4):
        fluorescent_beads_transformed[c] = affine_transform(fluorescent_beads[c], transform[c], order=3)

    # Add the images to napari
    colours = ["yellow", "red", "green", "blue"]
    for c in range(1, 4):
        viewer.add_image(
            fluorescent_beads[c],
            name="Camera " + str(cam_channels[c]),
            colormap=colours[c],
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            fluorescent_beads_transformed[c],
            name="Camera " + str(cam_channels[c]) + " transformed",
            colormap=colours[c],
            blending="additive",
            visible=True,
        )

    napari.run()


def view_bg_scale(nb: Notebook, t: int, r: int, c: int):
    """
    Visualize the background scaling for the selected tile, round, and channel.
    Args:
        nb: Notebook object (must have register and register_debug pages)
        t: tile to view
        r: round to view
        c: channel to view
    """
    assert nb.basic_info.use_preseq, "Background scaling is only available for pre-seq data."
    # get the data
    mid_z = len(nb.basic_info.use_z) // 2
    z_rad = 8
    z_range = np.arange(mid_z - z_rad, mid_z + z_rad + 1)
    r_pre = nb.basic_info.pre_seq_round
    bg_scale = nb.filter.bg_scale[t, r, c]
    # get the images
    base = load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r, c=c,
                      yxz=[None, None, z_range]).astype(np.float32)
    pre = load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r_pre, c=c,
                     yxz=[None, None, z_range]).astype(np.float32)
    affine_tr = nb.register.icp_correction[t, r, c].T
    affine_t_pre = nb.register.icp_correction[t, r_pre, c].T
    # change the shift as we are only looking at a subset of the z
    affine_tr[:, 3] += (affine_tr[:3, :3] - np.eye(3)) @ np.array([0, 0, mid_z - z_rad])
    affine_t_pre[:, 3] += (affine_t_pre[:3, :3] - np.eye(3)) @ np.array([0, 0, mid_z - z_rad])
    base = affine_transform(base, affine_tr, order=0)
    pre = affine_transform(pre, affine_t_pre, order=0)
    print("Images loaded and affine corrected.")
    flow_t_pre = np.load(os.path.join(nb.register.flow_dir, "smooth", f"t{t}_r{r_pre}.npy"), mmap_mode='r')[..., z_range]
    flow_t_pre = flow_t_pre.astype(np.float32)
    flow_t_r = np.load(os.path.join(nb.register.flow_dir, "smooth", f"t{t}_r{r}.npy"))[..., z_range]
    flow_t_r = flow_t_r.astype(np.float32)
    coords = np.array(np.meshgrid(range(base.shape[0]), range(base.shape[1]), range(base.shape[2]), indexing='ij'))
    print("Flows loaded.")
    warp_tr = coords - flow_t_r
    warp_t_pre = coords - flow_t_pre
    base = warp(base, warp_tr, order=0)[..., z_rad]
    pre = warp(pre, warp_t_pre, order=0)[..., z_rad]
    print("Images warped.")
    # blur base
    base = skimage.filters.gaussian(base, sigma=3)
    bright = pre > np.percentile(pre, 99)
    base_values = base[bright]
    pre_values = pre[bright]

    # create plots
    viewer = napari.Viewer()
    viewer.add_image(base, name=f't{t}_r{r}_c{c}', colormap="red", blending="additive")
    viewer.add_image(pre, name=f't{t}_r{r_pre}_c{c}', colormap="green", blending="additive")
    viewer.add_image(bright, name="bright", colormap="blue", blending="additive")

    # add the background scaling
    plt.scatter(x=pre_values, y=base_values, s=1, alpha=0.1)
    plt.plot(pre_values, bg_scale * pre_values, color="red", linestyle="--")
    plt.xlabel("pre values")
    plt.ylabel("base values")
    plt.title(f"Background scaling for t{t}, r{r}, c{c}")
    plt.show()


def view_overlay(nb: Notebook, t: int = None, rc1: tuple = None, rc2: tuple = None, use_z: np.ndarray = None):
    """
    Visualize the overlay of two images, both in the anchor frame of reference.
    Args:
        nb: Notebook object (must have register and register_debug pages)
        t: common tile
        rc1: (round, channel) for the first image
        rc2: (round, channel) for the second image
        use_z: list of z planes to load
    """
    assert rc1 is not None and rc2 is not None, "Please provide two (round, channel) pairs."
    if use_z is None:
        use_z = [z - 1 for z in nb.basic_info.use_z]
    new_origin = np.array([0, 0, use_z[0]])
    im = []
    # load the images
    for rc in [rc1, rc2]:
        r, c = rc
        suffix = "_raw" if r == nb.basic_info.pre_seq_round else ""
        im.append(load_image(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r, c=c,
                             yxz=[None, None, use_z], suffix=suffix).astype(np.float32))
    print('Images loaded.')

    ny, nx, nz = im[0].shape
    # apply the flow correction to both images
    for i, rc in enumerate([rc1, rc2]):
        r, c = rc
        # there is no flow correction for the anchor round, so skip
        if r == nb.basic_info.anchor_round:
            continue
        # load the flow
        flow = np.load(os.path.join(nb.register.flow_dir, "smooth", f"t{t}_r{r}.npy"), mmap_mode='r')[..., use_z]
        flow = flow.astype(np.float32)
        coords = np.meshgrid(range(ny), range(nx), range(nz), indexing='ij')
        # I think this should be a minus sign, as we are going from current round to anchor
        im[i] = warp(im[i], coords - flow, order=1)
    del coords, flow
    print('Images flow corrected.')

    # apply the affine correction (inverse) to both images
    for i, rc in enumerate([rc1, rc2]):
        r, c = rc
        # there is no affine transform for the anchor round
        if r == nb.basic_info.anchor_round:
            continue
        affine = nb.register.icp_correction[t, r, c]
        # there is no affine transform computed on the spots for the dapi channel (as there are no spots) but there is
        # an affine transform computed to correct the flow. Apply that if we are on the dapi images
        if c == nb.basic_info.dapi_channel:
            affine = nb.register_debug.round_correction[t, r]
        affine = preprocessing.adjust_affine(affine, new_origin)
        im[i] = affine_transform(im[i], affine, order=1, mode='constant', cval=0)
    print('Images affine corrected.')

    # create viewer
    viewer = napari.Viewer()
    colours = ["red", "green"]
    for i, rc in enumerate([rc1, rc2]):
        r, c = rc
        viewer.add_image(im[i], name=f't{t}_r{r}_c{c}', colormap=colours[i], blending="additive")
    viewer.dims.axis_labels = ['y', 'x', 'z']
    viewer.dims.order = (2, 0, 1)
    napari.run()


nb_file = '/home/reilly/local_datasets/dante_bad_trc_test/notebook.npz'
nb = Notebook(nb_file)
view_overlay(nb, t=4, rc1=(0, 14), rc2=(3, 14), use_z=np.arange(25, 35))