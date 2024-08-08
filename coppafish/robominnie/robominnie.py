import csv
import json
import math as maths
import os
import shutil
import time
from typing import Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Self

import dask
import napari
import numpy as np
import numpy.typing as npt
import pandas
import scipy
import tqdm

from .. import utils
from .. import log
from ..omp import base as omp_base
from ..pipeline import run
from ..setup.notebook import Notebook


# Originally created by Max Shinn, August 2023
# Refactored and expanded by Paul Shuker, September 2023 - present
class Robominnie:
    """
    Robominnie
    ==========
    Coppafish integration suite.

    Provides:
    ---------
    1. Modular, customisable synthetic data generation for coppafish
    2. Coppafish raw ``.npy`` file generation for full pipeline runs
    3. Coppafish is given an overall score using the ground-truth data

    Usage:
    ------
    Create new RoboMinnie instance for each integration test. Call functions for data generation (see ``robominnie.py``
    functions for options). Call ``save_raw_data``, then ``run_coppafish``.
    """

    # The index of the anchor channel in the images.
    anchor_channel: int = 1
    # The index of the dapi channel in all images.
    dapi_channel: int = 0
    # Robominnie creates one giant image which is then cookie-cut into tiles when saving the raw data for coppafish.
    # This variable gives the giant image edge padding which is useful when misaligning the images to test register.
    image_padding: Tuple[int, int, int] = (10, 10, 1)
    # The data type to keep images as while the synthetic data is being manipulated.
    image_dtype: np.dtype = np.float32
    invalid_tile_no: int = -10_000

    def __init__(
        self: Self,
        n_channels: int = 5,
        n_rounds: int = 7,
        n_planes: int = 4,
        tile_sz: int = 128,
        n_tiles_x: int = 1,
        n_tiles_y: int = 2,
        include_anchor: bool = True,
        include_dapi: bool = True,
        tile_overlap: float = 0.125,
        seed: Union[int, None] = 1,
    ) -> Self:
        """
        Create an empty Robominnie instance to begin generating synthetic data for coppafish.

        Args:
            - n_channels (int): the number of sequencing channels. Default: 5.
            - n_rounds (int): the number of sequencing rounds. Default: 7.
            - n_planes (int): the number of z planes. Default: 4.
            - tile_sz (int): the number of pixels along x/y directions for a single tile. Default: 128.
            - n_tiles_x (int): the number of tiles along the x direction. Default: 1.
            - n_tiles_y (int): the number of tiles along the y direction. Default: 2.
            - include_anchor (bool): include the anchor round and channel. Default: true.
            - include_dapi (bool): include the dapi channel. Default: true.
            - tile_overlap (float): the proportion of tile overlap in the x/y directions relative to the tile sizes.
                Default: 1/8 == 0.125.
            - seed (int or none): the random generation seed. None to use a random seed every time. Default: 1.
        """
        assert type(n_channels) is int
        assert n_channels > 1
        assert type(n_rounds) is int
        assert n_rounds > 1
        assert type(n_planes) is int
        assert n_planes >= 4
        assert type(tile_sz) is int
        assert tile_sz > 0
        assert type(n_tiles_x) is int
        assert n_tiles_x > 0
        assert type(n_tiles_y) is int
        assert n_tiles_y > 0
        assert type(include_anchor) is bool
        assert type(include_dapi) is bool
        assert type(tile_overlap) is float
        assert tile_overlap >= 0 and tile_overlap < 1
        assert seed is None or (type(seed) is int and seed >= 0)

        self.n_channels = n_channels
        self.use_channels = [i for i in range(self.dapi_channel + 1, self.dapi_channel + 1 + self.n_channels)]
        self.n_rounds = n_rounds
        self.n_planes = n_planes
        self.tile_sz = tile_sz
        self.n_tiles_x = n_tiles_x
        self.n_tiles_y = n_tiles_y
        self.n_tiles = self.n_tiles_x * self.n_tiles_y
        self.tile_overlap = tile_overlap
        self.include_anchor = include_anchor
        self.include_dapi = include_dapi
        self.rng = np.random.default_rng(seed)
        self.codes = None
        self.bleed_matrix = None
        # The spot positions are stored relative to the giant robominnie image throughout.
        self.true_spot_positions = np.zeros((0, 3), np.float32)
        self.true_spot_identities = np.zeros(0, str)
        self.true_spot_tile_numbers = np.zeros(0, np.int16)

        # NOTE: All 3D shapes are in the form y, x, and z.
        self.giant_image_shape = self._get_tile_bounds()[0][:, 1].max(0).astype(np.int16)
        self.giant_image_shape += np.array(self.image_padding, np.int16)
        self.giant_image_shape: Tuple[int, int, int] = tuple(self.giant_image_shape)
        sequence_images_shape = (self.n_rounds, self.n_channels + 1) + self.giant_image_shape
        self.sequence_images = np.zeros(sequence_images_shape, dtype=self.image_dtype)
        anchor_images_shape = (self.n_channels + 1,) + self.giant_image_shape
        self.anchor_image = np.zeros(anchor_images_shape, dtype=self.image_dtype)

    def generate_gene_codes(self, n_genes: int = 10) -> Dict[str, str]:
        """
        Generate random gene codes based on reed-solomon principle, using the lowest degree polynomial possible
        relative to the number of genes wanted to maximise the difference between codes. Saves the codes in self,
        which are then useable in function `add_spots`. The `i`th gene name will be `gene_i`. `ValueError` is raised
        if gene codes are already created.

        Args:
            n_genes (int, optional): number of unique gene codes to generate. Default: 12.

        Returns:
            Dict (str: str): gene names as keys, gene codes as values.

        Notes:
            See https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction for more details.
        """
        if self.codes is not None:
            raise ValueError(f"Already generated gene codes {len(self.codes)} gene codes")

        codes = utils.base.reed_solomon_codes(n_genes, self.n_rounds, self.n_channels)
        self.codes = codes

        return codes

    def generate_pink_noise(
        self,
        noise_amplitude: float = 1.5e-3,
        noise_spatial_scale: float = 0.1,
        include_sequence: bool = True,
        include_anchor: bool = True,
        include_dapi: bool = True,
    ) -> Self:
        """
        Superimpose pink noise onto images, if used. The noise is identical on all images because pink noise is a good
        estimation for biological things that fluoresce. See
        [here](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.814) for more details. You may expect the
        DAPI image to include pink noise that is not part of the other images because of the distinct nuclei staining.

        Args:
            noise_amplitude (float): The maximum possible noise intensity. Default: `0.0015`.
            noise_spatial_scale (float): Spatial scale of noise. Scales with image size. Default: `0.1`.
            include_sequence (bool, optional): Superimpose on sequencing images. Default: true.
            include_anchor (bool, optional): Superimpose on the anchor image. Default: true.
            include_dapi (bool, optional): Superimpose on the DAPI image. Default: true.
        """
        print(f"Generating pink noise")

        # True spatial scale should be maintained regardless of the image size, so we scale it as such.
        true_noise_spatial_scale = noise_spatial_scale * np.asarray(
            [*self.giant_image_shape[:2], 10 * self.giant_image_shape[2]]
        )
        # Generate pink noise
        pink_spectrum = 1 / (
            1
            + np.linspace(0, true_noise_spatial_scale[0], self.giant_image_shape[0])[:, None, None] ** 2
            + np.linspace(0, true_noise_spatial_scale[1], self.giant_image_shape[1])[None, :, None] ** 2
            + np.linspace(0, true_noise_spatial_scale[2], self.giant_image_shape[2])[None, None, :] ** 2
        )
        rand_pixels = self.rng.standard_normal(size=self.giant_image_shape, dtype=np.float32)
        pink_sampled_spectrum = pink_spectrum * np.fft.fftshift(scipy.fft.fftn(rand_pixels))
        pink_noise = np.abs(scipy.fft.ifftn(np.fft.ifftshift(pink_sampled_spectrum)))
        pink_noise = (pink_noise - np.mean(pink_noise)) * noise_amplitude / np.std(pink_noise)

        for r in range(self.n_rounds):
            for c in range(self.n_channels):
                if include_sequence:
                    self.sequence_images[r, c + 1 + self.dapi_channel] += pink_noise
            if include_dapi and self.include_dapi:
                self.sequence_images[r, self.dapi_channel] += pink_noise
        if include_anchor and self.include_anchor:
            self.anchor_image[self.anchor_channel] += pink_noise
        if include_dapi and self.include_dapi:
            self.anchor_image[self.dapi_channel] += pink_noise

    def add_spots(
        self,
        n_spots: Optional[int] = None,
        bleed_matrix: npt.NDArray[np.float_] = None,
        spot_size_pixels: npt.NDArray[np.float_] = None,
        spot_amplitude: float = 1,
        include_dapi: bool = False,
        spot_size_pixels_dapi: npt.NDArray[np.float_] = None,
        spot_amplitude_dapi: float = 1,
    ) -> Self:
        """
        Superimpose spots onto images in both space and channels (based on the bleed matrix). Also applied to the
        anchor when included. The spots are uniformly, randomly distributed across each image. We assume that
        `n_channels == n_dyes`.

        Args:
            n_spots (int, optional): Number of spots to superimpose. Default: `floor(0.2% * total_image_volume)`.
            bleed_matrix (`n_dyes x n_channels ndarray[float, float]`, optional): The bleed matrix, used to map each
                dye to its pattern as viewed by the camera in each channel. Default: Ones along the diagonals.
            spot_size_pixels (`(3) ndarray[float]`): The spot's standard deviation in directions `x, y, z`
                respectively. Default: `array([1.5, 1.5, 1.5])`.
            spot_amplitude (float, optional): Peak spot brightness scale factor. Default: `1`.
            include_dapi (bool, optional): Add spots to the DAPI channel in sequencing and anchor rounds, at the same
                positions. Default: false.
            spot_size_pixels_dapi (`(3) ndarray[float]`, optional): Spots' standard deviation when in the
                DAPI image. Default: Same as `spot_size_pixels`.
            spot_amplitude_dapi (float, optional): Peak DAPI spot brightness scale factor. Default: `1`.
        """

        def _blit(source, target, loc):
            """
            Superimpose given spot image (source) onto a target image (target) at the centred position loc. The
            parameter target is then updated with the final image.

            Args:
                source (n_channels (optional) x spot_size_y x spot_size_x x spot_size_z ndarray): The spot image.
                target (n_channels (optional) x tile_size_y x tile_size_x x tile_size_z ndarray): The tile image.
                loc (channel (optional), y, x, z ndarray): Central spot location.

            Returns:
                `(n_channels (optional) x tile_size_y x tile_size_x x tile_size_z) ndarray` target_blitted: tile image
                    with spots added.
            """
            source_size = np.asarray(source.shape)
            target_size = np.asarray(target.shape)
            # If we had infinite boundaries, where would we put it?  Assume "loc" is the centre of "target"
            target_loc_tl = loc - source_size // 2
            target_loc_br = target_loc_tl + source_size
            # Compute the index for the source
            source_loc_tl = -np.minimum(0, target_loc_tl)
            source_loc_br = source_size - np.maximum(0, target_loc_br - target_size)
            # Recompute the index for the target
            target_loc_br = np.minimum(target_size, target_loc_tl + source_size)
            target_loc_tl = np.maximum(0, target_loc_tl)
            # Compute slices from positions
            target_slices = [slice(s1, s2) for s1, s2 in zip(target_loc_tl, target_loc_br)]
            source_slices = [slice(s1, s2) for s1, s2 in zip(source_loc_tl, source_loc_br)]
            # Perform the blit
            target[tuple(target_slices)] += source[tuple(source_slices)]

            return target

        if bleed_matrix is None:
            bleed_matrix = np.diag(np.ones(self.n_channels))
        if spot_size_pixels is None:
            spot_size_pixels = np.asarray([1.5, 1.5, 1.5])
        assert (
            bleed_matrix.shape[1] == self.n_channels
        ), f"Bleed matrix does not have n_channels={self.n_channels} as expected"
        assert spot_size_pixels.shape[0] == 3, "`spot_size_pixels` must be in three dimensions"
        if bleed_matrix.shape[0] != bleed_matrix.shape[1]:
            log.warn(f"Given bleed matrix does not have equal channel and dye counts like usual")
        if self.bleed_matrix is None:
            self.bleed_matrix = bleed_matrix
        else:
            assert np.allclose(self.bleed_matrix, bleed_matrix), "All added spots must have the same bleed matrix"
        if spot_size_pixels_dapi is None:
            spot_size_pixels_dapi = spot_size_pixels.copy()
        assert spot_size_pixels_dapi.size == 3, "DAPI spot size must be in three dimensions"
        if n_spots is None:
            n_spots = maths.floor(0.2 * np.prod(self.giant_image_shape) / 100)
        assert n_spots > 0, f"Expected n_spots > 0, got {n_spots}"

        # Generate random spots.
        # Store the spots' global positions relative to the entire giant image. They are floats.
        true_spot_positions = self.rng.random(size=(n_spots, 3)) * list(self.giant_image_shape)
        true_spot_positions = true_spot_positions.astype(self.true_spot_positions.dtype)
        true_spot_identities = list(self.rng.choice(list(self.codes.keys()), n_spots))

        # We assume each spot is a multivariate Gaussian with a diagonal covariance,
        # where variance in each dimension is given by the spot size.  We create a spot
        # template image and then iterate through spots.  Each iteration, we add
        # ("blit") the spot onto the image such that the centre of the spot is in the
        # middle.  The size of the spot template is guaranteed to be odd, and is about
        # 1.5 times the standard deviation.  We add it to the appropriate colour channels
        # (by transforming through the bleed matrix) and then also add the spot to the
        # anchor.
        ind_size = np.ceil(spot_size_pixels * 1.5).astype(int) * 2 + 1
        indices = np.indices(ind_size) - ind_size[:, None, None, None] // 2
        spot_img = scipy.stats.multivariate_normal([0, 0, 0], np.eye(3) * spot_size_pixels).pdf(
            indices.transpose(1, 2, 3, 0)
        )
        np.multiply(spot_img, spot_amplitude * np.prod(spot_size_pixels) / 3.375, out=spot_img)
        ind_size = np.ceil(spot_size_pixels_dapi * 1.5).astype(int) * 2 + 1
        indices = np.indices(ind_size) - ind_size[:, None, None, None] // 2
        spot_img_dapi = scipy.stats.multivariate_normal([0, 0, 0], np.eye(3) * spot_size_pixels_dapi).pdf(
            indices.transpose(1, 2, 3, 0)
        )
        np.multiply(spot_img_dapi, spot_amplitude_dapi * np.prod(spot_size_pixels_dapi) / 3.375, out=spot_img_dapi)
        s = 0
        for p, ident in tqdm.tqdm(
            zip(true_spot_positions, true_spot_identities),
            desc="Superimposing spots",
            ascii=True,
            unit="spots",
            total=n_spots,
        ):
            p = np.asarray(p).astype(int)
            p_chan = np.round([self.n_channels // 2, p[0], p[1], p[2]]).astype(int)
            for r in range(self.n_rounds):
                dye = int(self.codes[ident][r])
                source = spot_img[None, :] * bleed_matrix[dye][:, None, None, None]
                self.sequence_images[r, self.use_channels] = _blit(
                    source, self.sequence_images[r, self.use_channels], p_chan
                )
                if include_dapi and self.include_dapi:
                    self.sequence_images[r, self.dapi_channel] = _blit(
                        spot_img_dapi, self.sequence_images[r, self.dapi_channel], p
                    )
            if self.include_anchor:
                self.anchor_image[self.anchor_channel] = _blit(spot_img, self.anchor_image[self.anchor_channel], p)
            if include_dapi and self.include_dapi and self.include_anchor:
                self.anchor_image[self.dapi_channel] = _blit(spot_img_dapi, self.anchor_image[self.dapi_channel], p)
            s += 1

        # Append just in case spots are superimposed multiple times
        if len(set(true_spot_identities)) != len(self.codes):
            log.warn("Some gene codes were never added, consider adding more spots")
        self.true_spot_identities = np.append(self.true_spot_identities, np.asarray(true_spot_identities))
        self.true_spot_positions = np.append(self.true_spot_positions, true_spot_positions, axis=0)
        # Spots will be assigned to tiles later when unstitching the images.
        self.true_spot_tile_numbers = np.full_like(self.true_spot_identities, self.invalid_tile_no, dtype=np.int16)

    # Post-Processing function
    def fit_images_to_type(self, type: np.dtype = np.uint16) -> Self:
        """
        Offset and multiply all used images by the same constant to fill the entire data range of the given datatype.

        Args:
            type (np.dtype, optional): datatype to scale images to. Must be integer. Default: uint16.
        """
        print(f"Scaling images to {np.dtype(type).name}...")

        type_min = np.iinfo(type).min
        type_max = np.iinfo(type).max
        image_min = self.sequence_images.min()
        image_max = self.sequence_images.max()
        if self.include_anchor or self.include_dapi:
            image_min = min(self.anchor_image.min(), image_min)
            image_max = max(self.anchor_image.max(), image_max)

        multiplier = (type_max - type_min) / (image_max - image_min)
        offset = type_max - multiplier * image_max
        assert np.isclose(offset, type_min - multiplier * image_min), f"Oops"
        self.sequence_images *= multiplier
        self.sequence_images += offset
        self.sequence_images = self.sequence_images.astype(type)
        if self.include_anchor or self.include_dapi:
            self.anchor_image *= multiplier
            self.anchor_image += offset
            self.anchor_image = self.anchor_image.astype(type)
        self.image_dtype = type

    def save_raw_images(self, output_dir: str, overwrite: bool = True) -> Self:
        """
        Save known spot positions and codes, raw .npy image files, metadata.json file, gene codebook and ``config.ini``
        file for coppafish pipeline run. Output directory must be empty. After saving, able to call function
        ``run_coppafish`` to run the coppafish pipeline.

        Args:
            output_dir (str): save directory.
            overwrite (bool, optional): overwrite any saved coppafish data inside the directory, delete old
                `notebook.npz` file if there is one and ignore any other files inside the directory. Default: true.
        """
        # Same dtype as ND2s
        self.fit_images_to_type()
        self.bound_spots()

        print(f"Saving raw data")

        self.tile_origins_yx, self.tile_yxz_pos = self._get_tile_bounds()[1:]
        self.image_tiles = self._unstitch_image(
            self.sequence_images, np.asarray([self.tile_sz, self.tile_sz, self.n_planes])
        )
        del self.sequence_images

        self.tile_xy_pos = self.tile_yxz_pos[:2]
        self.tilepos_yx_nd2 = self.tile_xy_pos

        if self.include_anchor:
            self.anchor_image_tiles = self._unstitch_image(
                self.anchor_image[None], np.asarray([self.tile_sz, self.tile_sz, self.n_planes])
            )
            self.anchor_image_tiles = self.anchor_image_tiles[0]
            del self.anchor_image

        if os.path.isdir(output_dir) and overwrite:
            shutil.rmtree(output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not overwrite:
            assert len(os.listdir(output_dir)) == 0, f"Output directory at \n\t{output_dir}\n must be empty"

        # Create an output_dir/output_coppafish directory for coppafish pipeline output saved to disk
        self.output = output_dir
        self.coppafish_output = os.path.join(output_dir, "output_coppafish")
        if overwrite:
            if os.path.isdir(self.coppafish_output):
                shutil.rmtree(self.coppafish_output)
        if not os.path.isdir(self.coppafish_output):
            os.mkdir(self.coppafish_output)

        # Create an output_dir/output_coppafish/tiles directory for coppafish extract output
        self.coppafish_tiles = os.path.join(self.coppafish_output, "tiles")
        if not os.path.isdir(self.coppafish_tiles):
            os.mkdir(self.coppafish_tiles)
        # Remove any old tile files in the tile directory, if any, to make sure coppafish runs extract and filter \
        # again
        for filename in os.listdir(self.coppafish_tiles):
            filepath = os.path.join(self.coppafish_tiles, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

        # Save the known gene names and positions to a csv.
        df = pandas.DataFrame(
            {
                "gene": self.true_spot_identities,
                "y": self.true_spot_positions[:, 0],
                "x": self.true_spot_positions[:, 1],
                "z": self.true_spot_positions[:, 2],
            }
        )
        df.to_csv(os.path.join(output_dir, "gene_locations.csv"))

        metadata = {
            "n_tiles": self.n_tiles,
            "n_rounds": self.n_rounds,
            "n_channels": self.n_channels + 1,
            "tile_sz": self.tile_sz,
            "pixel_size_xy": 0.26,
            "pixel_size_z": 0.9,
            "tile_centre": [self.tile_sz / 2, self.tile_sz / 2, self.n_planes / 2],
            "tilepos_yx": self.tile_origins_yx,
            "tilepos_yx_nd2": list(reversed(self.tile_origins_yx)),
            "channel_camera": [1] * (self.n_channels + 1),
            "channel_laser": [1] * (self.n_channels + 1),
            "xy_pos": self.tile_xy_pos,
            "nz": self.n_planes,
        }
        self.metadata_filepath = os.path.join(output_dir, "metadata.json")
        with open(self.metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save the raw .npy tile files, one round at a time, in separate round directories. We do this because
        # coppafish expects every rounds (including the anchor) in its own directory.
        # Dask saves each tile as a separate .npy file for coppafish to read.
        dask_chunks = (1, self.n_channels + 1, self.tile_sz, self.tile_sz, self.n_planes)
        for r in range(self.n_rounds):
            save_path = os.path.join(output_dir, f"{r}")
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            # Clear the raw .npy directories before dask saving, so old multi-tile data is not left in the
            # directories
            for filename in os.listdir(save_path):
                filepath = os.path.join(save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f"Found unexpected directory in {save_path}")
            image_dask = dask.array.from_array(self.image_tiles[r], chunks=dask_chunks)
            dask.array.to_npy_stack(save_path, image_dask)
            del image_dask
        if self.include_anchor:
            self.anchor_directory_name = f"anchor"
            anchor_save_path = os.path.join(output_dir, self.anchor_directory_name)
            if not os.path.isdir(anchor_save_path):
                os.mkdir(anchor_save_path)
            for filename in os.listdir(anchor_save_path):
                filepath = os.path.join(anchor_save_path, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                else:
                    raise IsADirectoryError(f"Unexpected subdirectory in {anchor_save_path}")
            image_dask = dask.array.from_array(self.anchor_image_tiles.astype(self.image_dtype), chunks=dask_chunks)
            dask.array.to_npy_stack(anchor_save_path, image_dask)
            del image_dask

        # Save the gene codebook in `output_dir`
        self.codebook_filepath = os.path.join(output_dir, "codebook.txt")
        with open(self.codebook_filepath, "w") as f:
            for gene_name, code in self.codes.items():
                f.write(f"{gene_name} {code}\n")

        # Save the gene colours in the output_dir, this is used for the coppafish `Viewer`.
        self.gene_colours_filepath = os.path.join(output_dir, "gene_colours.csv")
        with open(self.gene_colours_filepath, "w") as f:
            csvwriter = csv.writer(f, delimiter=",")
            napari_symbols = ["cross", "disc", "square", "triangle_up", "hbar", "vbar"]
            mpl_symbols = ["+", ".", "s", "^", "_", "|"]
            # Heading
            csvwriter.writerow(["", "GeneNames", "ColorR", "ColorG", "ColorB", "napari_symbol", "mpl_symbol"])
            for i, gene_name in enumerate(self.codes):
                random_index = self.rng.integers(len(napari_symbols))
                csvwriter.writerow(
                    [
                        f"{i}",
                        f"{gene_name}",
                        round(self.rng.random(), 2),
                        round(self.rng.random(), 2),
                        round(self.rng.random(), 2),
                        napari_symbols[random_index],
                        mpl_symbols[random_index],
                    ]
                )

        # Save the initial bleed matrix for the config file
        self.initial_bleed_matrix_filepath = os.path.join(output_dir, "bleed_matrix.npy")
        np.save(self.initial_bleed_matrix_filepath, self.bleed_matrix)

        # Add an extra channel and dye for the DAPI
        self.dye_names = map("".join, zip(["dye_"] * (self.n_channels), list(np.arange(self.n_channels).astype(str))))
        self.dye_names = list(self.dye_names)

        # Box sizes must be even numbers
        max_box_size_z, max_box_size_yx = 12, 300
        box_size_z = min([max_box_size_z, self.n_planes if self.n_planes % 2 == 0 else self.n_planes - 1])
        box_size_yx = min([max_box_size_yx, self.tile_sz if self.tile_sz % 2 == 0 else self.tile_sz - 1])

        # Save the config file. z_subvols is moved from the default of 5 based on n_planes.
        config_file_contents = f"""; This config file is auto-generated by RoboMinnie. 
        [file_names]
        input_dir = {output_dir}
        output_dir = {self.coppafish_output}
        tile_dir = {self.coppafish_tiles}
        initial_bleed_matrix = {self.initial_bleed_matrix_filepath}
        round = {', '.join([str(i) for i in range(self.n_rounds)])}
        anchor = {self.anchor_directory_name if self.include_anchor else ''}
        code_book = {self.codebook_filepath}
        raw_extension = .npy
        raw_metadata = {self.metadata_filepath}

        [basic_info]
        is_3d = true
        dye_names = {', '.join(self.dye_names)}
        use_rounds = {', '.join([str(i) for i in range(self.n_rounds)])}
        use_z = {', '.join([str(i) for i in range(self.n_planes)])}
        use_tiles = {', '.join(str(i) for i in range(self.n_tiles))}
        anchor_round = {self.n_rounds if self.include_anchor else ''}
        use_channels = {', '.join([str(i) for i in np.arange((self.dapi_channel + 1), (self.n_channels + 1))])}
        anchor_channel = {self.anchor_channel if self.include_anchor else ''}
        dapi_channel = {self.dapi_channel if self.include_dapi else ''}

        [extract]
        num_rotations = 0
        
        [filter]
        auto_thresh_multiplier = 2
        wiener_pad_shape = 40, 40, 9

        [find_spots]
        n_spots_warn_fraction = 0
        n_spots_error_fraction = 1

        [stitch]
        expected_overlap = {self.tile_overlap}

        [register]
        subvols = {self.n_planes}, {8}, {8}
        box_size = {box_size_z}, {box_size_yx}, {box_size_yx}
        pearson_r_thresh = 0.25
        round_registration_channel = {self.dapi_channel if (self.include_dapi) else ''}
        icp_min_spots = 10
        
        [call_spots]
        target_values = {", ".join(["1" for _ in range(self.n_channels)])}
        d_max = {", ".join(np.argmax(self.bleed_matrix, axis=1).astype(str))}
        
        [omp]
        max_genes = 10
        spot_shape = 13, 13, 1
        shape_isolation_distance_yx = 5
        pixel_max_percentile = 1
        shape_sign_thresh = 0.75
        score_threshold = 0.1
        subset_size_xy = 50
        """
        # Remove large spaces in the config contents
        config_file_contents = config_file_contents.replace("  ", "")

        self.config_filepath = os.path.join(output_dir, "robominnie.ini")
        with open(self.config_filepath, "w") as f:
            f.write(config_file_contents)

    def bound_spots(self) -> Self:
        """
        Remove true spot positions when they are not within a tile. Any spot positions within a tile overlap are added
        multiple times, one for each tile. This is called when save_raw_data is called.
        """
        print(f"Bounding spots")
        tile_bounds = self._get_tile_bounds()[0]

        bounded_true_spot_positions = np.zeros((0, 3), self.true_spot_positions.dtype)
        bounded_true_spot_identities = np.zeros(0, self.true_spot_identities.dtype)
        bounded_true_spot_tile_numbers = np.zeros(0, self.true_spot_tile_numbers.dtype)

        for t in range(self.n_tiles):
            in_tile = self.true_spot_positions >= tile_bounds[t, [0]]
            in_tile = np.logical_and(in_tile, self.true_spot_positions < tile_bounds[t, [1]])
            in_tile = in_tile.all(1)
            tile_spot_positions = self.true_spot_positions[in_tile]
            tile_spot_identities = self.true_spot_identities[in_tile]
            tile_spot_tile_numbers = np.full_like(tile_spot_identities, t, dtype=np.int16)
            bounded_true_spot_positions = np.append(bounded_true_spot_positions, tile_spot_positions, axis=0)
            bounded_true_spot_identities = np.append(bounded_true_spot_identities, tile_spot_identities)
            bounded_true_spot_tile_numbers = np.append(bounded_true_spot_tile_numbers, tile_spot_tile_numbers)

        print(f"{bounded_true_spot_positions.shape[0]} spots left out of {self.true_spot_positions.shape[0]}")
        self.true_spot_positions = bounded_true_spot_positions
        self.true_spot_identities = bounded_true_spot_identities
        self.true_spot_tile_numbers = bounded_true_spot_tile_numbers

    def run_coppafish(self) -> Notebook:
        """
        Run RoboMinnie instance on the entire coppafish pipeline.

        Returns:
            Notebook: final notebook.
        """
        print(f"Running coppafish")

        config_filepath = self.config_filepath
        n_planes = self.n_planes
        n_tiles = self.n_tiles

        start_time = time.time()
        nb = run.run_pipeline(config_filepath)

        # Keep the stitch information to convert local tiles coordinates into global coordinates when comparing
        # to true spots
        self.stitch_tile_origins = nb.stitch.tile_origin

        assert nb.has_page("stitch"), f"Stitch not found in notebook at {config_filepath}"

        # Keep reference spot information to compare to true spots, if wanted
        assert nb.has_page("ref_spots"), f"Reference spots not found in notebook at {config_filepath}"
        assert nb.has_page("call_spots")

        self.prob_spots_positions = nb.ref_spots.local_yxz.astype(np.float32)
        self.prob_spots_scores = nb.call_spots.gene_probabilities.max(1)
        self.prob_spots_gene_indices = np.argmax(nb.call_spots.gene_probabilities, 1)
        self.prob_spots_tile = nb.ref_spots.tile

        self.ref_spots_local_positions_yxz = nb.ref_spots.local_yxz.astype(np.float32)
        self.ref_spots_scores = nb.call_spots.dot_product_gene_score
        self.ref_spots_gene_indices = nb.call_spots.dot_product_gene_no
        self.ref_spots_tile = nb.ref_spots.tile

        end_time = time.time()
        print(
            f"Coppafish pipeline run: {round((end_time - start_time)/60, 1)}mins\n"
            + f"{round((end_time - start_time)//(n_planes * n_tiles), 1)}s per z plane per tile."
        )

        assert nb.has_page("omp"), f"OMP not found in notebook at {config_filepath}"
        # Keep the OMP spot intensities, assigned gene, assigned tile number and the spot positions in the robominnie
        # class
        self.omp_spot_scores = omp_base.get_all_scores(nb.basic_info, nb.omp)[0]
        self.omp_gene_numbers, self.omp_tile_number = omp_base.get_all_gene_no(nb.basic_info, nb.omp)
        self.omp_spot_local_positions = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)[0].astype(np.float32)
        self.omp_spot_count = self.omp_gene_numbers.shape[0]

        if self.omp_spot_count == 0:
            raise ValueError("Coppafish OMP found zero spots")

        return nb

    def score_tiles(self, method: Literal["prob", "anchor", "omp"], score_threshold: float = 0.0) -> Tuple[float]:
        """
        Computes the overall score as true positives / (true positives + wrong positives + false positives + false
        negatives) for each tile. This is done by comparing known spot locations to the spot locations from coppafish
        results. Only spots outside of the tile overlapping region are computed on so that duplicate spots are not an
        issue. See `coppafish/utils/errors.py` `compare_spots` function for details on how the scoring is computed.

        Args:
            - method (str): the method to compute for the score for. Can be 'OMP', 'anchor', or 'prob'.
            - score_threshold (float, optional): score threshold for spots being selected. Default: 0.

        Returns:
            (tuple of length `n_tiles`) overall_scores: each tile's overall score. The scores range from 0 to 100.
        """
        assert method in ("prob", "anchor", "omp")
        assert type(score_threshold) is float
        assert score_threshold >= 0

        if method == "prob":
            spot_positions = self.prob_spots_positions
            spot_tiles = self.prob_spots_tile
            spot_scores = self.prob_spots_scores
            spot_gene_indices = self.prob_spots_gene_indices
        elif method == "anchor":
            spot_positions = self.ref_spots_local_positions_yxz
            spot_tiles = self.ref_spots_tile
            spot_scores = self.ref_spots_scores
            spot_gene_indices = self.ref_spots_gene_indices
        elif method == "omp":
            spot_positions = self.omp_spot_local_positions
            spot_tiles = self.omp_tile_number
            spot_scores = self.omp_spot_scores
            spot_gene_indices = self.omp_gene_numbers
        else:
            raise ValueError(f"Unknown method: {method}")

        tile_origins = np.array(self._get_tile_bounds()[2]).astype(np.float32)
        tile_scores = []
        for t in range(self.n_tiles):
            in_tile = spot_tiles == t
            within_score_threshold = spot_scores >= score_threshold
            keep = in_tile & within_score_threshold
            t_spot_positions = spot_positions[keep]
            t_spot_gene_indices = spot_gene_indices[keep]
            # Cut out spot positions near the edge where there could be tile overlap.
            in_bound = self._is_not_overlapping(t_spot_positions)
            t_spot_positions = t_spot_positions[in_bound]
            t_spot_gene_indices = t_spot_gene_indices[in_bound]
            # Coppafish positions must be converted to global image positions for comparison.
            t_spot_positions += self.stitch_tile_origins[[t]]
            t_spot_positions += np.array(self.image_padding, np.float32)[np.newaxis]
            t_truth_positions = self.true_spot_positions[self.true_spot_tile_numbers == t]
            t_truth_gene_indices = self._get_true_gene_indices()[self.true_spot_tile_numbers == t]
            in_bound = self._is_not_overlapping(t_truth_positions - tile_origins[[t]])
            t_truth_positions = t_truth_positions[in_bound]
            t_truth_gene_indices = t_truth_gene_indices[in_bound]
            TPs, WPs, FPs, FNs = utils.errors.compare_spots(
                t_spot_positions, t_spot_gene_indices, t_truth_positions, t_truth_gene_indices, 2.0
            )
            t_score = 100 * TPs / (TPs + WPs + FPs + FNs)
            t_score = round(t_score * 10) / 10
            print(f"{TPs=}, {WPs=}, {FPs=}, {FNs=}")
            tile_scores.append(t_score)

        return tuple(tile_scores)

    def view_spot_positions(self) -> None:
        """
        View the true spot positions, coppafish's OMP, probability, and anchor spot positions in separate layers in
        a napari viewer. No visual distinction is made between genes.
        """
        colours = ("green", "blue", "red", "white")
        viewer = napari.Viewer()
        viewer.add_points(self.true_spot_positions, name="True", face_color=colours[0])
        viewer.add_points(
            self.prob_spots_positions + self.stitch_tile_origins[self.prob_spots_tile],
            name="Prob",
            face_color=colours[1],
        )
        viewer.add_points(
            self.ref_spots_local_positions_yxz + self.stitch_tile_origins[self.ref_spots_tile],
            name="Anchor",
            face_color=colours[2],
        )
        viewer.add_points(
            self.omp_spot_local_positions + self.stitch_tile_origins[self.omp_tile_number],
            name="OMP",
            face_color=colours[3],
        )
        napari.run()

    def _unstitch_image(
        self, image: npt.NDArray[np.float_], tile_size_yxz: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float_]:
        """
        Cookie-cut the large images into multiple tiles with a tile overlap.

        Args:
            image ((`n_rounds x n_channels x image_y x image_x x image_z`) `ndarray[float]`): giant image to separate
                into tiles.
            tile_size_yxz (`3` `ndarray[int]`): tile sizes.

        Returns:
            (`n_rounds x n_tiles x n_channels x tile_size_yxz[0] x tile_size_yxz[1] x tile_size_yxz[2]`)
                `ndarray[float]`: tile images, copy of `image`.
        """
        # TODO: Add support for affine-transformed tile images
        tile_images = np.zeros((image.shape[0], 0, image.shape[1], *tile_size_yxz), self.image_dtype)
        tile_bounds = self._get_tile_bounds()[0]
        t = 0
        for t in range(self.n_tiles):
            new_tile = image[
                :,
                None,
                :,
                tile_bounds[t, 0, 0] : tile_bounds[t, 1, 0],
                tile_bounds[t, 0, 1] : tile_bounds[t, 1, 1],
                tile_bounds[t, 0, 2] : tile_bounds[t, 1, 2],
            ]
            tile_images = np.append(tile_images, new_tile, axis=1)
            t += 1

        return tile_images

    def _get_tile_bounds(self) -> Tuple[np.ndarray[int], List[List[int]], List[List[float]]]:
        """
        Get each tile's minimum and maximum coordinates that bound the tile, relative to the giant image. Therefore,
        it is assumed each tile is a cuboid aligned with the y, x, and z axes.

        Returns:
            - (`(n_tiles x 2 x 3) ndarray[int]`) tile_bounds: tile_bounds[0, 0] is the minimum y, x, and z coordinate
                for the first tile (inclusive). tile_bounds[0, 1] is the maximum coordinate (exclusive).
            - (`list` of `list` of `int`): list of tile indices of form `[y_index, x_index]`.
            - (`list` of `list` of `float`): list of tile bottom-right corner positions, starting from `[0,0,0]`,
                in the form `[y,x,z]`.
        """
        tile_bounds = np.zeros((self.n_tiles, 2, 3), dtype=np.int16)
        tile_indices = []
        tile_positions_yxz = []
        t = 0
        for n_x in range(self.n_tiles_x):
            for n_y in range(self.n_tiles_y):
                tile_overlap_offset = -np.array(
                    [n_y * self.tile_sz * self.tile_overlap, n_x * self.tile_sz * self.tile_overlap, 0], np.float32
                )
                tile_overlap_offset = tile_overlap_offset.round(decimals=0).astype(np.int16)
                i_bound_min = np.array([n_y * self.tile_sz, n_x * self.tile_sz, 0], np.int16)
                i_bound_min += tile_overlap_offset
                i_bound_max = i_bound_min.copy() + np.array([self.tile_sz, self.tile_sz, self.n_planes], np.int16)
                tile_bounds[t, 0] = i_bound_min
                tile_bounds[t, 1] = i_bound_max
                tile_indices.append([n_y, n_x])
                tile_positions_yxz.append([n_y * self.tile_sz, n_x * self.tile_sz, 0.0])
                t += 1
        tile_bounds += np.array(self.image_padding, np.int16)[None, None]

        return tile_bounds, tile_indices, tile_positions_yxz

    def _get_true_gene_indices(self) -> np.ndarray[np.int16]:
        gene_names = list(self.codes.keys())
        gene_indices = [gene_names.index(name) for name in self.true_spot_identities.tolist()]

        return np.array(gene_indices, np.int16)

    def _is_not_overlapping(self, positions: np.ndarray) -> np.ndarray[bool]:
        assert positions.shape[1] == 3

        # Assuming that all z positions are not overlapping, since they should not be.
        not_overlapping = np.ones_like(positions, bool)
        not_overlapping[:, :2] = positions[:, :2] >= self.tile_sz * self.tile_overlap
        not_overlapping = not_overlapping[:, :2] & (positions[:, :2] <= self.tile_sz * (1 - self.tile_overlap))
        not_overlapping = not_overlapping.all(1)

        return not_overlapping
