import enum
import numbers
import os
from typing import Any, List, Optional, Tuple, Union

from numcodecs import Blosc, blosc
import numpy as np
import numpy.typing as npt
import numpy_indexed
import zarr

from .. import log, utils
from ..setup import NotebookPage


IMAGE_SAVE_DTYPE = np.uint16


class OptimisedFor(enum.Enum):
    FULL_READ_AND_WRITE = enum.auto()
    Z_PLANE_READ = enum.auto()


def get_compressor_and_chunks(
    optimised_for: OptimisedFor, image_shape: Tuple[int], image_z_index: Optional[int] = None
) -> Tuple[Any, Tuple[int]]:
    # By benchmarking every single blosc algorithm type (except snappy since this is unsupported by zarr),
    # chunk size and compression level, it was found that zstd, chunk size of 1x2x2 and compression level 3 was
    # fast at the sum of full image reading + writing times while still compressing the files to ~70-80%.
    # Benchmarking done by Paul Shuker (paul.shuker@outlook.com), January 2024.
    blosc.use_threads = True
    blosc.set_nthreads(utils.system.get_core_count())
    if image_z_index is None:
        image_z_index = np.argmin(image_shape).item()
    if optimised_for == OptimisedFor.FULL_READ_AND_WRITE:
        compressor = Blosc(cname="lz4", clevel=2, shuffle=Blosc.BITSHUFFLE)
        chunk_size_z = 1
        chunk_size_yx = 288
    elif optimised_for == OptimisedFor.Z_PLANE_READ:
        compressor = Blosc(cname="lz4", clevel=4, shuffle=Blosc.SHUFFLE)
        chunk_size_z = 1
        chunk_size_yx = 576
    else:
        raise ValueError(f"Unknown OptimisedFor value of {optimised_for}")
    if len(image_shape) >= 3:
        chunks = tuple()
        for i in range(len(image_shape)):
            if image_shape[i] < 10:
                chunks += (1,)
            elif i == image_z_index:
                chunks += (chunk_size_z,)
            else:
                chunks += (chunk_size_yx,)
    elif len(image_shape) == 2:
        chunks = (chunk_size_yx, chunk_size_yx)
    else:
        raise ValueError(f"Got image_shape with {len(image_shape)} dimensions: {image_shape}")
    return compressor, chunks


def get_pixel_max() -> int:
    """
    Get the maximum pixel value that can be saved in an image.
    """
    return np.iinfo(IMAGE_SAVE_DTYPE).max


def get_pixel_min() -> int:
    """
    Get the minimum pixel value that can be saved in an image.
    """
    return np.iinfo(IMAGE_SAVE_DTYPE).min


def image_exists(file_path: str, file_type: str) -> bool:
    """
    Checks if a tile exists at the given path locations.

    Args:
        file_path (str): tile path.
        file_type (str): file type.

    Returns:
        bool: tile existence.

    Raises:
        ValueError: unsupported file type.
    """
    if file_type.lower() == ".npy":
        return os.path.isfile(file_path)
    elif file_type.lower() == ".zarr":
        # Require a non-empty zarr directory
        return os.path.isdir(file_path) and len(os.listdir(file_path)) > 0
    else:
        log.error(ValueError(f"Unsupported file_type: {file_type.lower()}"))


def _save_image(
    image: npt.NDArray[np.uint16],
    file_path: str,
    file_type: str,
    optimised_for: OptimisedFor = None,
) -> None:
    """
    Save image in `file_path` location. No manipulation or logic is applied here, just purely saving the image as it
    was given.

    Args:
        image ((nz x ny x nx) ndarray[uint16]): image to save.
        file_path (str): file path.
        file_type (str): file type, case insensitive.
        optimised_for (literal[int]): what speed to optimise compression for. Affects the blosc compressor. Default:
            optimised for full image reading + writing time.

    Raises:
        ValueError: unsupported file_type or optimised_for.
    """
    if image.dtype != IMAGE_SAVE_DTYPE:
        raise ValueError(f"Expected image dtype {IMAGE_SAVE_DTYPE}, got {image.dtype}")

    if optimised_for is None:
        optimised_for = OptimisedFor.FULL_READ_AND_WRITE
    if file_type.lower() == ".npy":
        np.save(file_path, image)
    elif file_type.lower() == ".zarr":
        compressor, chunks = get_compressor_and_chunks(optimised_for, image.shape)
        zarray = zarr.open(
            store=file_path,
            shape=image.shape,
            mode="w",
            zarr_version=2,
            chunks=chunks,
            dtype="|u2",
            compressor=compressor,
        )
        zarray[:] = image
    else:
        raise ValueError(f"Unsupported `file_type`: {file_type.lower()}")


def _load_image(
    file_path: str, file_type: str, yxz: Optional[Tuple[Tuple[int], None]] = None
) -> npt.NDArray[np.uint16]:
    """
    Read in image from file_path location.

    Args:
        file_path (str): image location.
        file_type (str): file type. Either `'.npy'` or `'.zarr'`.
        yxz (tuple): a tuple of length 3. Contains tuples of length 2 and nones which specify the dimension minimum
            (inclusive) and maximum (exclusive) values to grab. Can be none to grab the entire dimension. For example,
            yxz=(None, (0, 25), None) would return all y values, x values from 0 to 24 (inclusive), and all z values
            to given an image of shape `(im_y x 25 x im_z)`. The image will always be returned with three-dimensions.
            Default: retrieve the entire image.

    Returns `(im_y x im_x x im_z) ndarray[uint16]`: loaded image.

    Raises:
        ValueError: unsupported file type.
    """
    if yxz is None:
        yxz = (None,) * 3
    assert type(yxz) is tuple
    assert len(yxz) == 3

    image = None
    if file_type.lower() == ".npy":
        image = np.load(file_path, mmap_mode=None)[:]
    elif file_type.lower() == ".zarr":
        image = zarr.open(file_path, mode="r")
    else:
        log.error(ValueError(f"Unsupported `file_type`: {file_type.lower()}"))

    shape_yxz = (image.shape[1], image.shape[2], image.shape[0])
    dim_indices_yxz = tuple()
    for dim, yxz_dim in enumerate(yxz):
        if yxz_dim is None:
            dim_indices_yxz += ((0, shape_yxz[dim]),)
            continue
        assert type(yxz_dim) is tuple, f"Expected tuple in yxz at index {dim} if not None, got {yxz_dim}"
        assert len(yxz_dim) == 2, f"Tuple must be length 2 inside of yxz, image min (inclusive) and max (exclusive)."
        assert type(yxz_dim[0]) is int and type(yxz_dim[1]) is int, f"Must be ints inside tuple in yxz, got {yxz_dim}"
        assert yxz_dim[0] < yxz_dim[1], f"The maximum must be greater than the minimum"
        assert yxz_dim[0] >= 0, f"The yxz minimum must be >= 0, got {yxz_dim[0]} for image dim {dim}"
        assert yxz_dim[1] <= shape_yxz[dim], f"The yxz maximum must be <= {shape_yxz[dim]}, got {yxz_dim[1]}"
        dim_indices_yxz += (yxz_dim,)
    image = image[
        dim_indices_yxz[2][0] : dim_indices_yxz[2][1],
        dim_indices_yxz[1][0] : dim_indices_yxz[1][1],
        dim_indices_yxz[0][0] : dim_indices_yxz[0][1],
    ]
    # zyx -> yxz.
    image = image.transpose((1, 2, 0))

    return image


def save_image(
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    file_type: str,
    image: npt.NDArray[np.int32],
    t: int,
    r: int,
    c: Optional[int] = None,
    suffix: str = "",
    apply_shift: bool = True,
    percent_clip_warn: float = None,
    percent_clip_error: float = None,
) -> npt.NDArray[np.uint16]:
    """
    Wrapper function to save tiles as npy files with correct shift. Moves z-axis to first axis before saving as it is
    quicker to load in this order. Tile `t` is saved to the path `nbp_file.tile[t,r,c]`, the path must contain an
    extension of `'.npy'`. The tile is saved as a `uint16`, so clipping may occur if the image contains really large
    values.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        file_type (str): the saving file type. Can be `'.npy'` or `'.zarr'`.
        image (`[ny x nx x nz] ndarray[int32]` or `[n_channels x ny x nx] ndarray[int32]`): image to save.
        t (int): npy tile index considering.
        r (int): round considering.
        c (int, optional): channel considering. Default: not given, raises error when `nbp_basic.is_3d == True`.
        suffix (str, optional): suffix to add to file name before the file extension. Default: empty.
        apply_shift (bool, optional): if true and saving a non-dapi channel, will apply the shift to the image.
        n_clip_warn (int, optional): if the number of pixels clipped off by saving is at least this number, a warning
            is logged. Default: never warn.
        n_clip_error (int, optional): if the number of pixels clipped off by saving is at least this number, then an
            error is raised. Default: never raise an error.

    Returns:
        `(nz x ny x nx) ndarray[uint16]`: the saved, manipulated image.
    """
    assert image.ndim == 3, "`image` must be 3 dimensional"

    if nbp_basic.is_3d:
        if c is None:
            log.error(ValueError("3d image but channel not given."))
        if not apply_shift or (c == nbp_basic.dapi_channel):
            # If dapi is given then image should already by uint16 so no clipping
            percent_clipped_pixels = (image < 0).sum()
            percent_clipped_pixels += (image > get_pixel_max()).sum()
            image = image.astype(IMAGE_SAVE_DTYPE)
        elif apply_shift and c != nbp_basic.dapi_channel:
            # need to shift and clip image so fits into uint16 dtype.
            # clip at 1 not 0 because 0 (or -tile_pixel_value_shift)
            # will be used as an invalid value when reading in spot_colors.
            percent_clipped_pixels = ((image + nbp_basic.tile_pixel_value_shift) < 1).sum()
            percent_clipped_pixels += ((image + nbp_basic.tile_pixel_value_shift) > get_pixel_max()).sum()
            image = np.clip(
                image + nbp_basic.tile_pixel_value_shift,
                1,
                get_pixel_max(),
                np.zeros_like(image, dtype=np.uint16),
                casting="unsafe",
            )
        percent_clipped_pixels *= 100 / image.size
        message = f"{t=}, {r=}, {c=} saved image has clipped {round(percent_clipped_pixels, 5)}% of pixels"
        if percent_clip_warn is not None and percent_clipped_pixels >= percent_clip_warn:
            log.warn(message)
        if percent_clip_error is not None and percent_clipped_pixels >= percent_clip_error:
            log.error(message)
        if percent_clipped_pixels >= 1:
            log.debug(message)
        # In 3D, cannot possibly save any un-used channel hence no exception for this case.
        expected_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
        if not utils.errors.check_shape(image, expected_shape):
            log.error(utils.errors.ShapeError("tile to be saved", image.shape, expected_shape))
        # yxz -> zxy
        image = np.swapaxes(image, 2, 0)
        # zxy -> zyx
        image = np.swapaxes(image, 1, 2)
        file_path = nbp_file.tile[t][r][c]
        file_path = file_path[: file_path.index(file_type)] + suffix + file_type
        _save_image(image, file_path, file_type, optimised_for=OptimisedFor.Z_PLANE_READ)
        return image
    else:
        log.error(NotImplementedError("2D image saving is currently not supported"))


def load_image(
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    file_type: str,
    t: int,
    r: int,
    c: int,
    yxz: Optional[Tuple[Tuple[int], None]] = None,
    apply_shift: bool = True,
    suffix: str = "",
) -> Union[npt.NDArray[np.uint16], npt.NDArray[np.int32]]:
    """
    Loads in image corresponding to desired tile, round and channel from the relevant npy file.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        file_type (str): the saved file type. Either `'.npy'` or `'.zarr'`.
        t (int): npy tile index considering.
        r (int): round considering.
        c (int): channel considering.
        yxz (tuple): a tuple of length 3. Contains tuples of length 2 and nones which specify the dimension minimum
            (inclusive) and maximum (exclusive) values to grab. Can be none to grab the entire dimension. For example,
            yxz=(None, (0, 25), None) would return all y values, x values from 0 to 24 (inclusive), and all z values
            to given an image of shape `(im_y x 25 x im_z)`. The image will always be returned with three-dimensions.
            Default: retrieve the entire image.
        apply_shift (bool, optional): if true and loading in a non-dapi channel, will apply the shift to the image to
            centre the zero correctly. This will convert the image from uint16 to int32.
        suffix (str, optional): suffix to add to file name to load from. Default: no suffix.

    Returns:
        (`(sz_y x sz_x x sz_z) ndarray`): loaded image.

    Notes:
        - May want to disable `apply_shift` to save memory as there will be no dtype conversion. If loading in DAPI,
            dtype is always `uint16` as there is no pixel shift.
    """
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_file) is NotebookPage
    assert type(file_type) is str
    assert type(t) is int, f"Got type {type(t)} instead"
    assert type(r) is int
    assert type(c) is int
    assert yxz is None or type(yxz) is tuple

    file_path = nbp_file.tile[t][r][c]
    file_path = file_path[: file_path.index(file_type)] + suffix + file_type

    if not image_exists(file_path, file_type):
        log.error(FileNotFoundError(f"Could not find image at {file_path} to load from"))

    # Image is in shape yxz.
    image = _load_image(file_path, file_type, yxz)

    # Apply shift if not DAPI channel
    if apply_shift and c != nbp_basic.dapi_channel:
        image = offset_pixels_by(image, -nbp_basic.tile_pixel_value_shift)
    return image


def get_npy_tile_ind(
    tile_ind_nd2: Union[int, List[int]], tile_pos_yx_nd2: np.ndarray, tile_pos_yx_npy: np.ndarray
) -> Union[int, List[int]]:
    """
    Gets index of tile in npy file from tile index of nd2 file.

    Args:
        tile_ind_nd2: Index of tile in nd2 file
        tile_pos_yx_nd2: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with nd2 index ```i```.
            Index 0 refers to ```YX = [0, 0]```.
            Index 1 refers to ```YX = [0, 1] if MaxX > 0```.
        tile_pos_yx_npy: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with npy index ```i```.
            Index 0 refers to ```YX = [MaxY, MaxX]```.
            Index 1 refers to ```YX = [MaxY, MaxX - 1] if MaxX > 0```.

    Returns:
        Corresponding indices in npy file.
    """
    if isinstance(tile_ind_nd2, numbers.Number):
        tile_ind_nd2 = [tile_ind_nd2]
    npy_index = numpy_indexed.indices(tile_pos_yx_npy, tile_pos_yx_nd2[tile_ind_nd2]).tolist()
    if len(npy_index) == 1:
        return npy_index[0]
    else:
        return npy_index


def offset_pixels_by(image: npt.NDArray[np.uint16], tile_pixel_value_shift: int) -> npt.NDArray[np.int32]:
    """
    Apply an integer, negative shift to every image pixel and convert datatype from uint16 to int32.

    Args:
        image (`ndarray[uint16]`): image to shift.
        tile_pixel_value_shift (int): shift.

    Returns:
        `ndarray[int32]`: shifted image.
    """
    assert tile_pixel_value_shift <= 0, "Cannot shift by a positive number"
    return image.astype(np.int32) + tile_pixel_value_shift
