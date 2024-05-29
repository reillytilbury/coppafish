import enum
import numbers
import os
from typing import Any, List, Optional, Tuple, Union

from numcodecs import Blosc, blosc
import numpy as np
import numpy.typing as npt
import numpy_indexed
from tqdm import tqdm
import zarr

from .. import extract, log, utils
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
        chunk_size_z = image_shape[image_z_index] // 2
        chunk_size_yx = min(288, np.max(image_shape).item())
    elif optimised_for == OptimisedFor.Z_PLANE_READ:
        compressor = Blosc(cname="lz4", clevel=4, shuffle=Blosc.SHUFFLE)
        chunk_size_z = image_shape[0] // 2
        chunk_size_yx = min(576, np.max(image_shape).item())
    else:
        raise ValueError(f"Unknown OptimisedFor value of {optimised_for}")
    if len(image_shape) >= 3:
        chunks = tuple()
        for i in range(len(image_shape)):
            if image_shape[i] < 10:
                chunks += (None,)
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
    file_path: str,
    file_type: str,
    indices: Optional[Union[Tuple[Union[List, int]], int]] = None,
    mmap_mode: str = None,
) -> npt.NDArray[np.uint16]:
    """
    Read in image from file_path location.

    Args:
        file_path (str): image location.
        file_type (str): file type. Either `'.npy'` or `'.zarr'`.
        indices (tuple or int, optional): coordinate indices to retrieve from the image. Default: entire image.
        mmap_mode (str, optional): the mmap_mode for numpy loading only. Default: no mapping.

    Returns `ndarray[uint16]`: loaded image.

    Raises:
        ValueError: unsupported file type.

    Notes:
        - For zarr, if indices is None then the entire image will be loaded into memory (not memory mapped like numpy),
            which can be slower if you only need a subset of the image.
        - Indexing a zarr array can be different from a numpy array, so we only support indexing with tuples and
            integers. See [here](https://zarr.readthedocs.io/en/stable/tutorial.html#advanced-indexing) for details.
    """
    if indices is None:
        indices = ...
    else:
        assert isinstance(indices, (int, tuple)), f"Unexpected indices type: {type(indices)}"

    if file_type.lower() == ".npy":
        return np.load(file_path, mmap_mode=mmap_mode)[indices]
    elif file_type.lower() == ".zarr":
        if indices == ...:
            return zarr.open(file_path, mode="r")[:]
        elif isinstance(indices, int):
            return zarr.open(file_path, mode="r")[indices, ...]
        return zarr.open(file_path, mode="r").get_coordinate_selection(indices)
    else:
        log.error(ValueError(f"Unsupported `file_type`: {file_type.lower()}"))


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
    yxz: Optional[Union[List, Tuple, np.ndarray]] = None,
    apply_shift: bool = True,
    suffix: str = "",
) -> npt.NDArray[Union[np.int32, np.uint16]]:
    """
    Loads in image corresponding to desired tile, round and channel from the relevant npy file.

    Args:
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        file_type (str): the saved file type. Either `'.npy'` or `'.zarr'`.
        t (int): npy tile index considering.
        r (int): round considering.
        c (int): channel considering.
        yxz (`list` of `int` or `ndarray[int]`, optional): if `None`, whole image is loaded otherwise there are two
            choices
            - `list` of `int [2 or 3]`. List containing y,x,z coordinates of sub image to load in.
                E.g. if `yxz = [np.array([5]), np.array([10,11,12]), np.array([8,9])]`
                returned `image` will have shape `[1 x 3 x 2]`.
                if `yxz = [None, None, z_planes]`, all pixels on given z_planes will be returned
                i.e. shape of image will be `[tile_sz x tile_sz x n_z_planes]`.
            - `[n_pixels x (2 or 3)] ndarray[int]`. Array containing yxz coordinates for which the pixel value is
                desired. E.g. if `yxz = np.ones((10,3))`, returned `image` will have shape `[10,]` with all values
                indicating the pixel value at `[1,1,1]`.
            Default: `None`.

        apply_shift (bool, optional): if true and loading in a non-dapi channel, will apply the shift to the image.
        suffix (str, optional): suffix to add to file name to load from. Default: no suffix.

    Returns:
        `int32 [ny x nx (x nz)]` or `int32 [n_pixels x (2 or 3)]`
            Loaded image.

    Notes:
        - May want to disable `apply_shift` to save memory and/or make loading quicker as there will be no dtype
            conversion. If loading in DAPI, dtype is always `uint16` as there is no shift.
    """
    if nbp_basic.is_3d:
        file_path = nbp_file.tile[t][r][c]
        file_path = file_path[: file_path.index(file_type)] + suffix + file_type
    else:
        log.error(NotImplementedError("2D image loading is currently not supported"))
    if not image_exists(file_path, file_type):
        log.error(FileNotFoundError(f"Could not find image at {file_path} to load from"))
    if yxz is not None:
        # Use mmap when only loading in part of image
        if isinstance(yxz, (list, tuple)):
            if nbp_basic.is_3d:
                if len(yxz) != 3:
                    log.error(ValueError(f"Loading in a 3D tile but dimension of coordinates given is {len(yxz)}."))
                if yxz[0] is None and yxz[1] is None:
                    z_indices = yxz[2]
                    if isinstance(z_indices, int):
                        image = _load_image(file_path, file_type, z_indices, mmap_mode="r")
                    else:
                        image = np.asarray(
                            [
                                _load_image(file_path, file_type, indices=int(z_indices[i]), mmap_mode="r")
                                for i in range(len(z_indices))
                            ],
                            dtype=np.uint16,
                        )
                    if image.ndim == 3:
                        # zyx -> yxz
                        image = np.moveaxis(image, 0, 2)
                else:
                    coord_index_zyx = np.ix_(yxz[2], yxz[0], yxz[1])
                    image = np.moveaxis(_load_image(file_path, file_type, indices=coord_index_zyx, mmap_mode="r"), 0, 2)
            else:
                if len(yxz) != 2:
                    log.error(ValueError(f"Loading in a 2D tile but dimension of coordinates given is {len(yxz)}."))
                coord_index = np.ix_(np.array([c]), yxz[0], yxz[1])  # add channel as first coordinate in 2D.
                # [0] below is to remove channel index of length 1.
                image = _load_image(nbp_file.tile[t][r], file_type, mmap_mode="r")[coord_index][0]
        elif isinstance(yxz, np.ndarray):
            if nbp_basic.is_3d:
                if yxz.shape[1] != 3:
                    log.error(ValueError(f"Loading in a 3D tile but dimension of coordinates given is {yxz.shape[1]}."))
                coord_index_zyx = tuple([yxz[:, j] for j in [2, 0, 1]])
                image = _load_image(file_path, file_type)[coord_index_zyx]
            else:
                if yxz.shape[1] != 2:
                    log.error(ValueError(f"Loading in a 2D tile but dimension of coordinates given is {yxz.shape[1]}."))
                coord_index = tuple(np.asarray(yxz[:, i]) for i in range(2))
                coord_index = (np.full(yxz.shape[0], c, int),) + coord_index  # add channel as first coordinate in 2D.
                image = _load_image(nbp_file.tile[t][r], file_type, mmap_mode="r")[coord_index]
        else:
            log.error(
                ValueError(
                    f"yxz should either be an [n_spots x n_dim] array to return an n_spots array indicating "
                    f"the value of the image at these coordinates or \n"
                    f"a list containing {2 + int(nbp_basic.is_3d)} arrays indicating the sub image to load."
                )
            )
    else:
        if nbp_basic.is_3d:
            # Don't use mmap when loading in whole image
            image = np.moveaxis(_load_image(file_path, file_type), 0, 2)
        else:
            # Use mmap when only loading in part of image
            image = _load_image(file_path, file_type, mmap_mode="r")[c]
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


def save_stitched(
    im_file: Union[str, None],
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_extract: NotebookPage,
    tile_origin: np.ndarray,
    r: int,
    c: int,
    from_raw: bool = False,
    zero_thresh: int = 0,
    num_rotations: int = 1,
) -> None:
    """
    Stitches together all tiles from round `r`, channel `c` and saves the resultant compressed npz at `im_file`. Saved
    image will be uint16 if from nd2 or from DAPI filtered npy files. Otherwise, if from filtered npy files, will
    remove shift and re-scale to fill int16 range.

    Args:
        im_file (str or none): path to save file. If `None`, stitched `image` is returned (with z axis last) instead of
            saved. Saved as a zarr array.
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_extract (NotebookPage): `extract` notebook page.
        tile_origin (`[n_tiles x 3] ndarray[float]`): yxz origin of each tile on round `r`.
        r (int): save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        c (int): save_stitched will save stitched image of all tiles of round `r`, channel `c`.
        from_raw (bool, optional): if `False`, will stitch together tiles from saved npy files, otherwise will load in
            raw un-filtered images from nd2/npy file. Default: false.
        zero_thresh (int, optional): all pixels with absolute value less than or equal to `zero_thresh` will be set to
            0. The larger it is, the smaller the compressed file will be. Default: 0.
        num_rotations (int, optional): the number of rotations to apply to each tile individually. Default: `1`, the
            same as the notebook default.
    """
    yx_origin = np.round(tile_origin[:, :2]).astype(int)
    z_origin = np.round(tile_origin[:, 2]).astype(int).flatten()
    yx_size = np.max(yx_origin, axis=0) + nbp_basic.tile_sz
    if nbp_basic.is_3d:
        z_size = z_origin.max() + nbp_basic.nz
        stitched_image = np.zeros(np.append(z_size, yx_size), dtype=np.uint16)
    else:
        z_size = 1
        stitched_image = np.zeros(yx_size, dtype=np.uint16)
    if from_raw:
        round_dask_array, _ = utils.raw.load_dask(nbp_file, nbp_basic, r=r)
        shift = 0  # if from nd2 file, data type is already un-shifted uint16
    else:
        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
            shift = 0  # if filtered dapi, data type is already un-shifted uint16
        else:
            # if from filtered npy files, data type is shifted uint16, want to save stitched as un-shifted int16.
            shift = nbp_basic.tile_pixel_value_shift
    if shift != 0:
        # change dtype to accommodate negative values and set base value to be zero in the shifted image.
        stitched_image = stitched_image.astype(np.int32) + shift
    with tqdm(total=z_size * len(nbp_basic.use_tiles), desc="Saving stitched image") as pbar:
        for t in nbp_basic.use_tiles:
            if from_raw:
                (image_t,) = utils.raw.load_image(nbp_file, nbp_basic, t, c, round_dask_array, r, list(nbp_basic.use_z))
                # replicate non-filtering procedure in extract_and_filter
                if not nbp_basic.is_3d:
                    image_t = extract.focus_stack(image_t)
                image_t, bad_columns = extract.strip_hack(image_t)  # find faulty columns
                image_t[:, bad_columns] = 0
                if nbp_basic.is_3d:
                    image_t = np.moveaxis(image_t, 2, 0)  # put z-axis back to the start
                if num_rotations != 0:
                    image_t = np.rot90(image_t, k=num_rotations, axes=(1, 2))
            else:
                if nbp_basic.is_3d:
                    image_t = load_image(
                        nbp_file, nbp_basic, nbp_extract.file_type, t, r, c, apply_shift=True
                    ).transpose((2, 0, 1))
                else:
                    image_t = load_image(nbp_file, nbp_basic, nbp_extract.file_type, t, r, c, apply_shift=False)
            for z in range(z_size):
                # any tiles not used will be kept as 0.
                pbar.set_postfix({"tile": t, "z": z})
                if nbp_basic.is_3d:
                    file_z = z - z_origin[t]
                    if file_z < 0 or file_z >= len(nbp_basic.use_z):
                        # Set tile to 0 if currently outside its area
                        local_image = np.zeros((nbp_basic.tile_sz, nbp_basic.tile_sz))
                    else:
                        local_image = image_t[file_z]
                    stitched_image[
                        z,
                        yx_origin[t, 0] : yx_origin[t, 0] + nbp_basic.tile_sz,
                        yx_origin[t, 1] : yx_origin[t, 1] + nbp_basic.tile_sz,
                    ] = local_image
                else:
                    stitched_image[
                        yx_origin[t, 0] : yx_origin[t, 0] + nbp_basic.tile_sz,
                        yx_origin[t, 1] : yx_origin[t, 1] + nbp_basic.tile_sz,
                    ] = image_t
                pbar.update(1)
    pbar.close()
    if shift != 0:
        # remove shift and re-scale so fits the whole int16 range
        # Break things up by z plane so that not everything needs to be stored in ram at once
        im_max = np.abs(stitched_image).max()
        for z in range(stitched_image.shape[0]):
            stitched_image[z] = stitched_image[z] - shift
            stitched_image[z] = stitched_image[z] * np.iinfo(np.int16).max / im_max

        stitched_image = np.rint(stitched_image, np.zeros_like(stitched_image, dtype=np.int16), casting="unsafe")
    if zero_thresh > 0:
        stitched_image[np.abs(stitched_image) <= zero_thresh] = 0

    if im_file is None:
        if z_size > 1:
            stitched_image = np.moveaxis(stitched_image, 0, -1)
        return stitched_image
    else:
        zarray = zarr.open_array(im_file, mode="w", shape=stitched_image.shape, dtype=stitched_image.dtype)
        zarray[:] = stitched_image


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
