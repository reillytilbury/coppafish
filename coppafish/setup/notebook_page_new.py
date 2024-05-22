import os
import time
import json
import shutil
from typing import Any, Dict, List, Tuple

import zarr
import numpy as np

from .. import utils


class NotebookPage:
    def get_page_name(self):
        return self._name

    _name: str
    name = property(get_page_name)
    # Attribute names allowed to be set inside the notebook page that are not in _options.
    _VALID_ATTRIBUTE_NAMES = ("_name", "_time_created", "_version")

    _metadata_name: str = "_metadata.json"

    _page_name_key: str = "page_name"
    _time_created: float
    _time_created_key: str = "time_created"
    _version: str
    _version_key: str = "version"

    # Each page variable is given a list. The list contains a datatype(s) in the first index followed by a description.
    # A variable can be allowed to take multiple datatypes by separating them with an ' or '. Check the supported
    # types by looking at the function _is_types at the end of this file. The 'tuple' is a special datatype that can be
    # nested. For example, tuple[tuple[int]] is a valid datatype.
    _datatype_separator: str = " or "
    _datatype_nest_start: str = "["
    _datatype_nest_end: str = "]"
    _options: Dict[str, Dict[str, list]] = {
        "basic_info": {
            "anchor_channel": ["int or none", "Channel in anchor used. None if anchor not used."],
            "anchor_round": [
                "int or none",
                "Index of anchor round (typically the first round after imaging rounds so `anchor_round = n_rounds`)."
                + "`None` if anchor not used.",
            ],
            "dapi_channel": [
                "int or none",
                "Channel in anchor round that contains *DAPI* images. `None` if no *DAPI*.",
            ],
            "use_channels": ["tuple[int]", "n_use_channels. Channels in imaging rounds to use throughout pipeline."],
            "use_rounds": ["tuple[int]", "n_use_rounds. Imaging rounds to use throughout pipeline."],
            "use_z": ["tuple[int]", "z planes used to make tile *npy* files"],
            "use_tiles": [
                "tuple[int]",
                "n_use_tiles tiles to use throughout pipeline."
                + "For an experiment where the tiles are arranged in a $4 \\times 3$ ($n_y \\times n_x$) grid, "
                + "tile indices are indicated as below:"
                + "\n"
                + "| 2  | 1  | 0  |"
                + "\n"
                + "| 5  | 4  | 3  |"
                + "\n"
                + "| 8  | 7  | 6  |"
                + "\n"
                + "| 11 | 10 | 9  |",
            ],
            "use_dyes": ["tuple[int]", "n_use_dyes dyes to use when assigning spots to genes."],
            "dye_names": [
                "tuple[str] or none",
                "Names of all dyes so for gene with code $360...$,"
                + "gene appears with `dye_names[3]` in round $0$, `dye_names[6]` in round $1$, `dye_names[0]` in round $2$ etc."
                + "`none` if each channel corresponds to a different dye.",
            ],
            "is_3d": [
                "bool",
                "`True` if *3D* pipeline used, `False` if *2D*",
            ],
            "channel_camera": [
                "tuple[int] or none",
                "`channel_camera[i]` is the wavelength in *nm* of the camera on channel $i$."
                + "`none` if `dye_names = none`.",
            ],
            "channel_laser": [
                "tuple[int] or none",
                "`channel_laser[i]` is the wavelength in *nm* of the laser on channel $i$."
                + "`none` if `dye_names = none`.",
            ],
            "tile_pixel_value_shift": [
                "int",
                "This is added onto every tile (except *DAPI*) when it is saved and removed from every tile when loaded."
                + "Required so we can have negative pixel values when save to *npy* as *uint16*."
                + "*Typical=15000*",
            ],
            "n_extra_rounds": [
                "int",
                "Number of non-imaging rounds, typically 1 if using anchor and 0 if not.",
            ],
            "n_rounds": [
                "int",
                "Number of imaging rounds in the raw data",
            ],
            "tile_sz": [
                "int",
                "$yx$ dimension of tiles in pixels",
            ],
            "n_tiles": [
                "int",
                "Number of tiles in the raw data",
            ],
            "n_channels": [
                "int",
                "Number of channels in the raw data",
            ],
            "nz": [
                "int",
                "Number of z-planes used to make the *npy* tile images (can be different from number in raw data).",
            ],
            "n_dyes": [
                "int",
                "Number of dyes used",
            ],
            "tile_centre": [
                "ndarray[float]",
                "`[y, x, z]` location of tile centre in units of `[yx_pixels, yx_pixels, z_pixels]`."
                + "For *2D* pipeline, `tile_centre[2] = 0`",
            ],
            "tilepos_yx_nd2": [
                "ndarray[int]",
                "[n_tiles x 2] `tilepos_yx_nd2[i, :]` is the $yx$ position of tile with *fov* index $i$ in the *nd2* file."
                + "Index 0 refers to `YX = [0, 0]`"
                + "Index 1 refers to `YX = [0, 1]` if `MaxX > 0`",
            ],
            "tilepos_yx": [
                "ndarray[int]",
                "[n_tiles x 2] `tilepos_yx[i, :]` is the $yx$ position of tile with tile directory (*npy* files) index $i$."
                + "Equally, `tilepos_yx[use_tiles[i], :]` is $yx$ position of tile `use_tiles[i]`."
                + "Index 0 refers to `YX = [MaxY, MaxX]`"
                + "Index 1 refers to `YX = [MaxY, MaxX - 1]` if `MaxX > 0`",
            ],
            "pixel_size_xy": [
                "float",
                "$yx$ pixel size in microns",
            ],
            "pixel_size_z": [
                "float",
                "$z$ pixel size in microns",
            ],
            "use_anchor": [
                "bool",
                "whether or not to use anchor",
            ],
            "use_preseq": [
                "bool",
                "whether or not to use pre-seq round",
            ],
            "pre_seq_round": [
                "int or none",
                "round number of pre-seq round",
            ],
            "bad_trc": [
                "tuple[tuple[int]]",
                "Tuple of bad tile, round, channel combinations. If a tile, round, channel combination is in this,"
                + "it will not be used in the pipeline.",
            ],
        },
        "file_names": {
            "input_dir": [
                "dir",
                "Where raw *nd2* files are",
            ],
            "output_dir": [
                "dir",
                "Where notebook is saved",
            ],
            "tile_dir": [
                "dir",
                "Where filtered image files are saved, from both extract and filter sections of the pipeline",
            ],
            "tile_unfiltered_dir": [
                "dir",
                "Where extract, unfiltered image files are saved",
            ],
            "round": [
                "tuple[file]",
                "n_rounds names of *nd2* files for the imaging rounds. If not using, will be an empty list.",
            ],
            "anchor": [
                "str or none",
                "Name of *nd2* file for the anchor round. `none` if anchor not used",
            ],
            "raw_extension": [
                "str",
                "*.nd2* or *.npy* indicating the data type of the raw data.",
            ],
            "raw_metadata": [
                "str or none",
                "If `raw_extension = .npy`, this is the name of the *json* file in `input_dir` which contains the "
                + "required metadata extracted from the initial *nd2* files."
                + "I.e. it is the output of *coppafish/utils/nd2/save_metadata*",
            ],
            "dye_camera_laser": [
                "file",
                "*csv* file giving the approximate raw intensity for each dye with each camera/laser combination",
            ],
            "code_book": [
                "file",
                "Text file which contains the codes indicating which dye to expect on each round for each gene",
            ],
            "scale": [
                "file",
                "Text file saved containing the `extract['scale']` and `extract['scale_anchor']` values used to create "
                + "the tile *npy* files in the *tile_dir*. If the second value is 0, it means `extract['scale_anchor']` "
                + "has not been calculated yet."
                + ""
                + "If the extract step of the pipeline is re-run with `extract['scale']` or "
                + "`extract['scale_anchor']` different to values saved here, an error will be raised.",
            ],
            "spot_details_info": [
                "file",
                "After each tile is finished in find_spots, information about spots found is saved as array to *npz* file:"
                + "We save this info as an npz because we're saving a collection of arrays. We have an int16 array [n_spots x 3]"
                + "containing $y$, $x$, $z$, called spot_details. We have an int16 array called spot_no [n_tiles * n_rounds * "
                + "n_channels] which gives number of spots found on that [t,r,c]. We also have an [n_anchor_spots * 1] boolean"
                + "array named isolated_spots which gives 1 if anchor_spot[s] isolated, 0 o/w",
            ],
            "psf": [
                "file",
                "*npy* file location indicating the average spot shape" + "This will have the shape `n_z x n_y x n_x`.",
            ],
            "big_dapi_image": [
                "file or none",
                "*npz* file of stitched *DAPI* image. none if `nb.basic_info.dapi_channel = none`"
                + "If *3D*, 1st axis in *npz* file is z.",
            ],
            "big_anchor_image": [
                "file",
                "*npz* file of stitched image of `ref_round`/`ref_channel`. Will be stitched anchor if anchor used."
                + "If *3D*, 1st axis in *npz* file is z.",
            ],
            "pciseq": [
                "tuple[file]",
                "2 *csv* files where plotting information for *pciSeq* is saved."
                + "\n"
                + "`pciseq[0]` is the path where the *OMP* method output will be saved."
                + "\n"
                + "`pciseq[1]` is the path where the *ref_spots* method output will be saved."
                + "\n"
                + "If files don't exist, they will be created when the function *coppafish/export_to_pciseq* is run.",
            ],
            "tile": [
                "tuple[tuple[tuple[file]]]",
                "List of string arrays `n_tiles x (n_rounds + n_extra_rounds) x n_channels` if 3d}]"
                + "`tile[t][r][c]` is the [extract][file_type] filtered file containing all z planes for tile $t$, "
                + "round $r$, channel $c$",
            ],
            "tile_unfiltered": [
                "tuple[tuple[tuple[file]]]",
                "List of string arrays [n_tiles][(n_rounds + n_extra_rounds) {x n_channels if 3d}]"
                + "`tile[t][r][c]` is the [extract][file_type] unfiltered file containing all z planes for tile $t$, "
                + "round $r$, channel $c$",
            ],
            "fluorescent_bead_path": [
                "str or none",
                "Path to *nd2* file containing fluorescent beads. `none` if not used.",
            ],
            "pre_seq": [
                "str or none",
                "Name of *nd2* file for the pre-sequencing round. `none` if not used",
            ],
            "initial_bleed_matrix": [
                "dir or none",
                "Location of initial bleed matrix file. If `none`, then use the default bleed matrix",
            ],
        },
        "extract": {
            "file_type": [
                "str",
                "File type used to save tiles after extraction.",
            ],
        },
        "filter": {
            "auto_thresh": [
                "ndarray[int]",
                "Numpy int array `[n_tiles x (n_rounds + n_extra_rounds) x n_channels]`"
                + "`auto_thresh[t, r, c]` is the threshold spot intensity for tile $t$, round $r$, channel $c$"
                + "used for spot detection in the `find_spots` step of the pipeline.",
            ],
            "image_scale": [
                "float",
                "Every non-DAPI image is scaled by this number after filtering. It is computed using the first non-DAPI "
                + "image in filter. The scaling helps to use more of the uint16 integer range when saving the images for "
                + "improved pixel value precision.",
            ],
            "bg_scale": [
                "ndarray[float]",
                "Numpy float array `[n_tiles x n_rounds x n_channels]`"
                + "`bg_scale[t, r, c]` is the scale factor applied to the preseq round of tile $t$, channel $c$"
                + "to match the colour profile of the sequencing image in tile t, round r, channel c."
                + "None if not using preseq round.",
            ],
        },
        "filter_debug": {
            "r_dapi": [
                "int or none",
                "Filtering for *DAPI* images is a tophat with `r_dapi` radius."
                + "Should be approx radius of object of interest."
                + "Typically this is 8 micron converted to yx-pixel units which is typically 48."
                + "By default, it is `None` meaning *DAPI* not filtered at all and *npy* file not saved.",
            ],
            "psf": [
                "ndarray[float]",
                "Numpy float array [psf_shape[0] x psf_shape[1] x psf_shape[2]] or None (psf_shape is in config file)"
                + "Average shape of spot from individual raw spot images normalised so max is 1 and min is 0."
                + "`None` if not applying the Wiener deconvolution.",
            ],
            "z_info": [
                "int",
                "z plane in *npy* file from which `auto_thresh` and `hist_counts` were calculated. By default, this is "
                + "the mid plane.",
            ],
            "invalid_auto_thresh": [
                "int",
                "Any `filter.auto_thresh` value set to this is invalid.",
            ],
            "time_taken": [
                "float",
                "Time taken to run through the filter section, in seconds.",
            ],
        },
        "find_spots": {
            "isolation_thresh": [
                "ndarray[float]",
                "Numpy float array [n_tiles]"
                + "Spots found on tile $t$, `ref_round`, `ref_channel` are isolated if annular filtered image"
                + "is below `isolation_thresh[t]` at spot location."
                + "\n"
                + "*Typical: 0*",
            ],
            "spot_no": [
                "ndarray[int]",
                "Numpy int32 array [n_tiles x (n_rounds + n_extra_rounds) x n_channels]"
                + "`spot_no[t, r, c]` is the number of spots found on tile $t$, round $r$, channel $c$",
            ],
            "spot_yxz": [
                "ndarray[int]",
                "Numpy int16 array [n_total_spots x 3]"
                + "`spot_yxz[i,:]` is `[y, x, z]` for spot $i$"
                + "$y$, $x$ gives the local tile coordinates in yx-pixels. "
                + "$z$ gives local tile coordinate in z-pixels (0 if *2D*)",
            ],
            "isolated_spots": [
                "ndarray[bool]",
                "Boolean Array [n_anchor_spots x 1]"
                + "isolated spots[s] returns a 1 if anchor spot s is isolated and 0 o/w",
            ],
        },
        "stitch": {
            "tile_origin": [
                "ndarray[float]",
                "Numpy float array [n_tiles x 3]"
                + "`tile_origin[t,:]` is the bottom left $yxz$ coordinate of tile $t$."
                + "$yx$ coordinates in yx-pixels and z coordinate in z-pixels."
                + "nan is populated in places where a tile is not used in the pipeline.",
            ],
            "north_start_shift_search": [
                "ndarray[int]",
                "Numpy integer array [3 x 3]"
                + "Initial search range used to find overlap between north neighbouring tiles"
                + "`[i, :]` is the `[min, max, step]` of the search in direction $i$ (0 is $y$, 1 is $x$, 2 is $z$)."
                + "`[2,:]` is in units of z-pixels and is 0 if *2D*.",
            ],
            "east_start_shift_search": [
                "ndarray[int]",
                "Numpy integer array [3 x 3]"
                + "Initial search range used to find overlap between east neighbouring tiles"
                + "`[i, :]` is the `[min, max, step]` of the search in direction $i$ (0 is $y$, 1 is $x$, 2 is $z$)."
                + "`[2,:]` is in units of z-pixels and is 0 if *2D*.",
            ],
            "north_final_shift_search": [
                "ndarray[int]",
                "Numpy integer array [3 x 3]"
                + "Final search range used to find overlap between north neighbouring tiles"
                + "`[i, :]` is the `[min, max, step]` of the search in direction $i$ (0 is $y$, 1 is $x$, 2 is $z$)."
                + "`[2,:]` is in units of z-pixels and is 0 if *2D*.",
            ],
            "east_final_shift_search": [
                "ndarray[int]",
                "Numpy integer array [3 x 3]"
                + "Final search range used to find overlap between east neighbouring tiles"
                + "`[i, :]` is the `[min, max, step]` of the search in direction $i$ (0 is $y$, 1 is $x$, 2 is $z$)."
                + "`[2,:]` is in units of z-pixels and is 0 if *2D*.",
            ],
            "north_pairs": [
                "ndarray[int]",
                "Numpy integer array [n_north_overlap x 2]"
                + "`north_pairs[i, 1]` is the tile to the north of `north_pairs[i, 0]`",
            ],
            "east_pairs": [
                "ndarray[int]",
                "Numpy integer array [n_east_overlap x 2]"
                + "`east_pairs[i, 1]` is the tile to the east of `east_pairs[i, 0]`",
            ],
            "north_shifts": [
                "ndarray[int]",
                "Numpy integer array [n_north_overlap x 3]"
                + "`north_shifts[i, :]` is the $yxz$ shift found that is applied to `north_pairs[i, 0]` to take it to `north_pairs[i, 1]`"
                + ""
                + "Units: `[yx_pixels, yx_pixels, z_pixels]`, `[:, 2] = 0` if *2D*.",
            ],
            "east_shifts": [
                "ndarray[int]",
                "Numpy integer array [n_east_overlap x 3]"
                + "`east_shifts[i, :]` is the $yxz$ shift found that is applied to `east_pairs[i, 0]` to take it to `east_pairs[i, 1]`"
                + ""
                + "Units: `[yx_pixels, yx_pixels, z_pixels]`, `[:, 2] = 0` if *2D*.",
            ],
            "north_score": [
                "ndarray[float]",
                "Numpy float array [n_north_overlap x 1]"
                + "`north_score[i]` is approximately the number of matches found for `north_shifts[i, :]`",
            ],
            "east_score": [
                "ndarray[float]",
                "Numpy float array [n_east_overlap x 1]"
                + "`east_score[i]` is approximately the number of matches found for `east_shifts[i, :]`",
            ],
            "north_score_thresh": [
                "ndarray[float]",
                "Numpy float array [n_north_overlap x 1]"
                + "If `north_score[i]` is below `north_score_thresh[i]`, it indicates `north_shifts[i]` may be incorrect.",
            ],
            "east_score_thresh": [
                "ndarray[float]",
                "Numpy float array [n_east_overlap x 1]"
                + "If `east_score[i]` is below `east_score_thresh[i]`, it indicates `east_shifts[i]` found may be incorrect.",
            ],
            "north_outlier_shifts": [
                "ndarray[int]",
                "Numpy integer array [n_north_overlap x 3]"
                + "If `north_score[i]` was below `north_score_thresh[i]`, `north_shifts[i]` was found again"
                + "and old shift recorded as `north_outlier_shifts[i]`. Will be zero if this did not happen.",
            ],
            "east_outlier_shifts": [
                "ndarray[int]",
                "Numpy integer array [n_east_overlap x 3]"
                + "If `east_score[i]` was below `east_score_thresh[i]`, `east_shifts[i]` was found again"
                + "and old shift recorded as `east_outlier_shifts[i]`. Will be zero if this did not happen.",
            ],
            "north_outlier_score": [
                "ndarray[float]",
                "Numpy float array [n_north_overlap x 1]"
                + "If `north_score[i]` was below `north_score_thresh[i]`, `north_shifts[i]` was found again"
                + "and old score recorded as `north_outlier_score[i]`. Will be zero if this did not happen.",
            ],
            "east_outlier_score": [
                "ndarray[float]",
                "Numpy float array [n_east_overlap x 1]"
                + "If `east_score[i]` was below `east_score_thresh[i]`, `east_shifts[i]` was found again"
                + "and old score recorded as `east_outlier_score[i]`. Will be zero if this did not happen.",
            ],
        },
        "register": {
            "flow_dir": [
                "str",
                "Directory where the flow files are saved",
            ],
            "icp_correction": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_rounds x n_channels x 4 x 3]"
                + "yxz affine corrections to be applied after the warp.",
            ],
        },
        "register_debug": {
            "channel_transform_initial": [
                "ndarray[float]",
                "Numpy float array [n_channels x 4 x 3]"
                + "Initial affine transform to go from the ref round/channel to each imaging channel.",
            ],
            "round_correction": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_rounds x 4 x 3]"
                + "yxz affine corrections to be applied after the warp.",
            ],
            "channel_correction": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_channels x 4 x 3]"
                + "yxz affine corrections to be applied after the warp.",
            ],
            "n_matches_round": [
                "ndarray[int]",
                "Numpy int array [n_tiles x n_rounds x n_icp_iters]"
                + "Number of matches found for each iteration of icp for the round correction.",
            ],
            "mse_round": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_rounds x n_icp_iters]"
                + "Mean squared error for each iteration of icp for the round correction.",
            ],
            "converged_round": [
                "ndarray[bool]",
                "Numpy boolean array [n_tiles x n_rounds]"
                + "Whether the icp algorithm converged for the round correction.",
            ],
            "n_matches_channel": [
                "ndarray[int]",
                "Numpy int array [n_tiles x n_channels x n_icp_iters]"
                + "Number of matches found for each iteration of icp for the channel correction.",
            ],
            "mse_channel": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_channels x n_icp_iters]"
                + "Mean squared error for each iteration of icp for the channel correction.",
            ],
            "converged_channel": [
                "ndarray[bool]",
                "Numpy boolean array [n_tiles x n_channels]"
                + "Whether the icp algorithm converged for the channel correction.",
            ],
        },
        "ref_spots": {
            "local_yxz": [
                "ndarray[int]",
                "Numpy int16 array [n_spots x 3]. "
                + "`local_yxz[s]` are the $yxz$ coordinates of spot $s$ found on `tile[s]`, `ref_round`, `ref_channel`."
                + "To get `global_yxz`, add `nb.stitch.tile_origin[tile[s]]`.",
            ],
            "isolated": [
                "ndarray[bool]",
                "Numpy boolean array [n_spots]. "
                + "`True` for spots that are well isolated i.e. surroundings have low intensity so no nearby spots.",
            ],
            "tile": [
                "ndarray[int]",
                "Numpy int16 array [n_spots]. Tile each spot was found on.",
            ],
            "colors": [
                "ndarray[int]",
                "Numpy int32 array [n_spots x n_rounds x n_channels]. "
                + "`[s, r, c]` is the intensity of spot $s$ on round $r$, channel $c$."
                + "`-tile_pixel_value_shift` if that round/channel not used otherwise integer.",
            ],
            "gene_no": [
                "ndarray[int]",
                "Numpy int16 array [n_spots]. Gene number assigned to each spot. `None` if not assigned.",
            ],
            "score": [
                "ndarray[float]",
                "Numpy float array [n_spots]. `score[s]' is the highest gene coef of spot s.",
            ],
            "score_diff": [
                "ndarray[float]",
                "Numpy float array [n_spots]. "
                + "`score_diff[s]` is the difference between the highest and second highest gene score of spot s.",
            ],
            "intensity": [
                "ndarray[float]",
                "Numpy float32 array [n_spots]. "
                + "$\\chi_s = \\underset{r}{\\mathrm{median}}(\\max_c\\zeta_{s_{rc}})$"
                + "where $\\pmb{\\zeta}_s=$ `colors[s, r]/color_norm_factor[r]`.",
            ],
            "background_strength": [
                "ndarray[float]",
                "Numpy float32 array [n_spots x n_channels_use]"
                + "Background strength of each spot. Calculated as the median of the channel across rounds.",
            ],
            "gene_probs": [
                "ndarray[float]",
                "Numpy float array [n_spots x n_genes]. Von-Mises probability that spot $s$ is gene $g$.",
            ],
            "dye_strengths": [
                "ndarray[float]",
                "Numpy float array [n_spots x n_rounds x n_dyes]. Strength of each dye in each round for each spot.",
            ],
            "bg_colours": [
                "ndarray[int]",
                "Numpy int32 array [n_spots x n_rounds x n_channels]. "
                + "Background colour of each spot. Calculated as the median of the channel across rounds.",
            ],
        },
        "call_spots": {
            "gene_names": [
                "ndarray[str]",
                "Numpy string array [n_genes]" + "Names of all genes in the code book provided.",
            ],
            "gene_codes": [
                "ndarray[int]",
                "Numpy integer array [n_genes x n_rounds]"
                + "`gene_codes[g, r]` indicates the dye that should be present for gene $g$ in round $r$.",
            ],
            "color_norm_factor": [
                "ndarray[float]",
                "Numpy float array [n_tiles x n_rounds x n_channels]"
                + "Normalisation such that dividing `colors` by `color_norm_factor` equalizes intensity of channels."
                + "`config['call_spots']['bleed_matrix_method']` indicates whether normalisation is for rounds and channels or just channels.",
            ],
            "abs_intensity_percentile": [
                "ndarray[float]",
                "Numpy float array [100]]" + "Percentile of `intensity` to use for thresholding in omp.",
            ],
            "initial_bleed_matrix": [
                "ndarray[float]",
                "Numpy float array [n_channels x n_dyes]"
                + "Starting point for determination of bleed matrix."
                + "If separate dye for each channel, `initial_bleed_matrix[r]` will be the identity matrix for each $r$."
                + "Otherwise, it will be `initial_raw_bleed_matrix` divided by `color_norm_factor`.",
            ],
            "bleed_matrix": [
                "ndarray[float]",
                "Numpy float array [n_channels x n_dyes]"
                + "For a spot, $s$, which should be dye $d$ in round $r$, we expect `color[s, r]/color_norm_factor[r]`"
                + "to be a constant multiple of `bleed_matrix[r, :, d]`",
            ],
            "bled_codes": [
                "ndarray[float]",
                "Numpy float array [n_genes x n_rounds x n_channels]"
                + "`color[s, r]/color_norm_factor[r]` of spot, $s$, corresponding to gene $g$"
                + "is expected to be a constant multiple of `bled_codes[g, r]` in round $r$."
                + "`nan` if $r$/$c$ outside `use_rounds`/`use_channels` and 0 if `gene_codes[g,r]` outside `use_dyes`."
                + "All codes have L2 norm = 1 when summed across all `use_rounds` and `use_channels`.",
            ],
            "bled_codes_ge": [
                "ndarray[float]",
                "Numpy float array [n_genes x n_rounds x n_channels]"
                + "`color[s, r]/color_norm_factor[r]` of spot, $s$, corresponding to gene $g$"
                + "is expected to be a constant multiple of `bled_codes[g, r]` in round $r$."
                + "`nan` if $r$/$c$ outside `use_rounds`/`use_channels` and 0 if `gene_codes[g,r]` outside `use_dyes`."
                + "All codes have L2 norm = 1 when summed across all `use_rounds` and `use_channels`.",
            ],
            "gene_efficiency": [
                "ndarray[float]",
                "Numpy float array [n_genes x n_rounds]"
                + "`gene_efficiency[g,r]` gives the expected intensity of gene $g$ in round $r$ compared to that expected by the `bleed_matrix`."
                + "It is computed based on the average of isolated spot_colors assigned to that gene"
                + "which exceed `score`, `score_diff` and `intensity` thresholds given in config file."
                + "For all $g$, there is an `av_round[g]` such that `gene_efficiency[g, av_round[g]] = 1`."
                + "`nan` if $r$ outside `use_rounds` and 1 if `gene_codes[g,r]` outside `use_dyes`.",
            ],
            "use_ge": [
                "ndarray[bool]",
                "Bool [n_spots]. Mask for all spots used in gene efficiency calculation.",
            ],
        },
        "omp": {
            "spot_tile": [
                "int",
                "`spot` was found from isolated spots detected on this tile.",
            ],
            "mean_spot": [
                "ndarray[float]",
                "Numpy float16 array [shape_max_size[0] x shape_max_size[1] x shape_max_size[2]] or None"
                + "Mean of *OMP* coefficient sign in neighbourhood centred on detected isolated spot.",
            ],
            "spot": [
                "ndarray[int]",
                "Numpy integer array [shape_size_y x shape_size_x x shape_size_z]"
                + "Expected sign of *OMP* coefficient in neighbourhood centered on spot."
                + ""
                + "1 means expected positive coefficient."
                + ""
                + "0 means unsure of sign.",
            ],
            "local_yxz": [
                "ndarray[int]",
                "Numpy int16 array [n_spots, 3]"
                + "`local_yxz[s]` are the $yxz$ coordinates of spot $s$ found on `tile[s]`, `ref_round`, `ref_channel`."
                + "To get `global_yxz`, add `nb.stitch.tile_origin[tile[s]]`.",
            ],
            "scores": [
                "ndarray[int]",
                "Numpy int16 array [n_spots]"
                + "For each spot `s`, specified by position `local_yxz[s]` at tile `tile[s]` with gene read `gene_no[s]`, "
                + "has gene read score of `scores[s]`. Each score is between 0 and `np.iinfo(np.int16).max`.",
            ],
            "tile": [
                "ndarray[int]",
                "Numpy int16 array [n_spots]" + "Tile each spot was found on.",
            ],
            "gene_no": [
                "ndarray[int]",
                "Numpy int16 array [n_spots]" + "`gene_no[s]` is the index of the gene assigned to spot $s$.",
            ],
            "colours": [
                "ndarray[int]",
                "Numpy int32 `(n_spots x len(use_rounds) x len(use_channels))`"
                + "Each spot's intensity in every sequencing round/channel before colour normalisation.",
            ],
        },
        "thresholds": {
            "intensity": [
                "float",
                "Final accepted reference and OMP spots require `intensity > thresholds[intensity]`."
                + "This is copied from `config[thresholds]` and if not given there, will be set to "
                + "`nb.call_spots.gene_efficiency_intensity_thresh`."
                + "intensity for a really intense spot is about 1 so intensity_thresh should be less than this.",
            ],
            "score_ref": [
                "float",
                "Final accepted reference spots are those which pass `quality_threshold` which is:"
                + ""
                + "`nb.ref_spots.score > thresholds[score_ref]` and `intensity > thresholds[intensity]`."
                + ""
                + "This is copied from `config[thresholds]`."
                + "Max score is 1 so `score_ref` should be less than this.",
            ],
            "score_omp": [
                "float",
                "Final accepted *OMP* spots are those which pass `quality_threshold` which is:"
                + ""
                + "`score > thresholds[score_omp]` and `intensity > thresholds[intensity]`."
                + ""
                + "`score` is given by:"
                + ""
                + "`score = (score_omp_multiplier * n_neighbours_pos + n_neighbours_neg) / "
                + "(score_omp_multiplier * n_neighbours_pos_max + n_neighbours_neg_max)`."
                + ""
                + "This is copied from `config[thresholds]`."
                + "Max score is 1 so `score_thresh` should be less than this.",
            ],
            "score_omp_multiplier": [
                "float",
                "Final accepted OMP spots are those which pass quality_threshold which is:"
                + ""
                + "`score > thresholds[score_omp]` and `intensity > thresholds[intensity]`."
                + ""
                + "score is given by:"
                + ""
                + "`score = (score_omp_multiplier * n_neighbours_pos + n_neighbours_neg) / "
                + "(score_omp_multiplier * n_neighbours_pos_max + n_neighbours_neg_max)`."
                + ""
                + "This is copied from `config[thresholds]`.",
            ],
        },
        # For unit testing only.
        "debug": {
            "a": ["int"],
            "b": ["float"],
            "c": ["bool"],
            "d": ["tuple[int]"],
            "e": ["tuple[tuple[float]]"],
            "f": ["int or float"],
            "g": ["none"],
            "h": ["float or none"],
            "i": ["str"],
            "j": ["ndarray[float]"],
            "k": ["ndarray[int]"],
            "l": ["ndarray[bool]"],
            "m": ["ndarray[str]"],
            "n": ["ndarray[uint]"],
            "o": ["zarr"],
        },
    }
    _type_prefixes: Dict[str, str] = {
        "int": "json",
        "float": "json",
        "str": "json",
        "bool": "json",
        "file": "json",
        "dir": "json",
        "tuple": "json",
        "none": "json",
        "ndarray": "npz",
        "zarr": "zarr",
    }

    def __init__(self, page_name: str) -> None:
        if page_name not in self._options.keys():
            raise ValueError(f"Could not find _options for page called {page_name}")
        self._name = page_name
        self._time_created = time.time()
        self._version = utils.system.get_software_version()
        self._sanity_check_options()

    def save(self, page_directory: str, /) -> None:
        """
        Save the notebook page to the given directory. If the directory already exists, do not overwrite it.
        """
        if os.path.isdir(page_directory):
            return
        if len(self.get_unset_variables()) > 0:
            raise ValueError(
                f"Cannot save unfinished page {self._name}. "
                + f"Variable(s) {self._get_unset_variables()} not assigned yet."
            )

        os.mkdir(page_directory)
        metadata_path = self._get_metadata_path(page_directory)
        self._save_metadata(metadata_path)
        for name in self._options[self._name].keys():
            value = self.__getattribute__(name)
            types_as_str: str = self._options[self._name][name][0]
            self._save_variable(name, value, types_as_str, page_directory)

    def load(self, page_directory: str, /) -> None:
        """
        Load all variables from inside the given directory. All variables already set inside of the page are
        overwritten.
        """
        if not os.path.isdir(page_directory):
            raise FileNotFoundError(f"Could not find page directory at {page_directory} to load from")

        metadata_path = self._get_metadata_path(page_directory)
        self._load_metadata(metadata_path)
        for name in self._options[self._name].keys():
            self.__setattr__(name, self._load_variable(name, page_directory))

    def get_unset_variables(self) -> Tuple[str]:
        """
        Return a tuple of all variable names that have not been set to a valid value in the notebook page.
        """
        unset_variables = []
        for variable_name in self._options[self._name].keys():
            try:
                self.__getattribute__(variable_name)
            except AttributeError:
                unset_variables.append(variable_name)
        return tuple(unset_variables)

    def __gt__(self, variable_name: str) -> None:
        """
        Print a variable's description by doing `notebook_page > "variable_name"`.
        """
        assert type(variable_name) is str
        if variable_name not in self._options[self._name].keys():
            print(f"No variable named {variable_name}")
            return
        if len(self._options[self._name][variable_name]) <= 1:
            print(f"No description found for {variable_name}")

        print(f"{self._options[self._name][variable_name][1]}")

    def __setattr__(self, name: str, value: Any, /) -> None:
        """
        Deals with syntax `notebook_page.name = value`.
        """
        if name in self._VALID_ATTRIBUTE_NAMES:
            object.__setattr__(self, name, value)
            return

        if name not in self._options[self._name].keys():
            raise NameError(f"Cannot set variable {name} in {self._name} page. It is not inside _options")
        expected_types = self._get_expected_types(name)
        if not self._is_types(value, expected_types):
            raise TypeError(f"Failed to set variable {name} to type {type(value)}. Expected type(s) {expected_types}")

        object.__setattr__(self, name, value)

    def _save_metadata(self, file_path: str) -> None:
        assert not os.path.isfile(file_path), f"Metadata file at {file_path} should not exist"

        metadata = {
            self._page_name_key: self._name,
            self._time_created_key: self._time_created,
            self._version_key: self._version,
        }
        with open(file_path, "x") as file:
            file.write(json.dumps(metadata, indent=4))

    def _load_metadata(self, file_path: str) -> None:
        assert os.path.isfile(file_path), f"Metadata file at {file_path} not found"

        metadata: dict = None
        with open(file_path, "r") as file:
            metadata = json.loads(file.read())
            assert type(metadata) is dict
        self._name = metadata[self._page_name_key]
        self._time_created = metadata[self._time_created_key]
        self._version = metadata[self._version_key]

    def _get_metadata_path(self, page_directory: str) -> str:
        return os.path.join(page_directory, self._metadata_name)

    def _get_page_directory(self, in_directory: str) -> str:
        return os.path.join(in_directory, self._name)

    def _get_expected_types(self, name: str) -> str:
        return self._options[self._name][name][0]

    def _save_variable(self, name: str, value: Any, types_as_str: str, page_directory: str) -> None:
        file_prefix = self._type_str_to_prefix(types_as_str.split(self._datatype_separator)[0])
        file_path = os.path.join(page_directory, f"{name}.{file_prefix}")

        if file_prefix == "json":
            with open(file_path, "x") as file:
                file.write(json.dumps({"value": value}, indent=4))
        elif file_prefix == "npz":
            value.setflags(write=False)
            np.savez_compressed(file_path, value)
        elif file_prefix == "zarr":
            if not os.path.isdir(value):
                raise FileNotFoundError(f"Failed to find zarr at {value}")
            value_zarr: zarr.Array = zarr.open_array(store=value, mode="r")
            if type(value_zarr) is not zarr.Array:
                raise TypeError(f"File at {file_path} was of type {type(value_zarr)}, expected zarr array")
            saved_value = zarr.open_array(
                store=file_path,
                mode="w",
                shape=value_zarr.shape,
                dtype=value_zarr.dtype,
                order=value_zarr.order,
                compressor=value_zarr.compressor,
                chunks=value_zarr.chunks,
                zarr_version=2,
            )
            saved_value[:] = value_zarr[:]
            saved_value.read_only = True
            if os.path.normpath(value) != os.path.normpath(file_path):
                # Delete the old location of the zarr array.
                shutil.rmtree(value)
            self.__setattr__(name, file_path)
        else:
            raise NotImplementedError(f"File prefix {file_prefix} is not supported")

    def _load_variable(self, name: str, page_directory: str) -> Any:
        types_as_str = self._options[self._name][name][0].split(self._datatype_separator)
        file_prefix = self._type_str_to_prefix(types_as_str[0])
        file_path = os.path.join(page_directory, f"{name}.{file_prefix}")

        if file_prefix == "json":
            with open(file_path, "r") as file:
                value = json.loads(file.read())["value"]
            return value
        elif file_prefix == "npz":
            return np.load(file_path)["arr_0"]
        elif file_prefix == "zarr":
            return file_path
        else:
            raise NotImplementedError(f"File prefix {file_prefix} is not supported")

    def _sanity_check_options(self) -> None:
        # Only multiple datatypes can be options for the same variable if they save to the same save file type. So, a
        # variable's type cannot be "ndarray[int] or zarr" because they save into different file types.
        for page_name, page_options in self._options.items():
            for var_name, var_list in page_options.items():
                unique_prefixes = set()
                types_as_str = var_list[0]
                for type_as_str in types_as_str.split(self._datatype_separator):
                    unique_prefixes.add(self._type_str_to_prefix(type_as_str))
                if len(unique_prefixes) > 1:
                    raise TypeError(
                        f"Variable {var_name} in page {page_name} has incompatible types: "
                        + f"{' and '.join(unique_prefixes)} in _options"
                    )

    def _type_str_to_prefix(self, type_as_str: str) -> str:
        return self._type_prefixes[type_as_str.split(self._datatype_nest_start)[0]]

    def _is_types(self, value: Any, types_as_str: str) -> bool:
        valid_types: List[str] = types_as_str.split(self._datatype_separator)
        for type_str in valid_types:
            if self._is_type(value, type_str):
                return True
        return False

    def _is_type(self, value: Any, type_as_str: str) -> bool:
        if self._datatype_separator in type_as_str:
            raise ValueError(f"Type {type_as_str} in _options cannot contain the phrase {self._datatype_separator}")

        if type_as_str == "none":
            return value is None
        elif type_as_str == "int":
            return type(value) is int
        elif type_as_str == "float":
            return type(value) is float
        elif type_as_str == "str":
            return type(value) is str
        elif type_as_str == "bool":
            return type(value) is bool
        elif type_as_str == "file":
            return type(value) is str
        elif type_as_str == "dir":
            return type(value) is str
        elif type_as_str == "tuple":
            return type(value) is tuple
        elif type_as_str.startswith("tuple"):
            if not type(value) is tuple:
                return False
            if len(value) == 0:
                return True
            else:
                for subvalue in value:
                    if not self._is_type(
                        subvalue, type_as_str[len("tuple" + self._datatype_nest_start) : -len(self._datatype_nest_end)]
                    ):
                        return False
                return True
        elif type_as_str == "ndarray[int]":
            return type(value) is np.ndarray and isinstance(value.dtype.type(), (np.int16, np.int32, np.int64))
        elif type_as_str == "ndarray[uint]":
            return type(value) is np.ndarray and isinstance(value.dtype.type(), (np.uint16, np.uint32, np.uint64))
        elif type_as_str == "ndarray[float]":
            return type(value) is np.ndarray and isinstance(value.dtype.type(), (np.float16, np.float32, np.float64))
        elif type_as_str == "ndarray[str]":
            return type(value) is np.ndarray and isinstance(value.dtype.type(), (str, np.str_))
        elif type_as_str == "ndarray[bool]":
            return type(value) is np.ndarray and isinstance(value.dtype.type(), (np.bool_, bool))
        elif type_as_str == "zarr":
            # A zarr is specified by pointing the notebook page to the location of the zarr array. This way the array
            # does not need to be in memory all the time and the array can be deleted once it is saved within the
            # notebook page.
            return type(value) is str and os.path.isdir(value) and value.endswith(".zarr")
        else:
            raise TypeError(f"Unexpected type '{type_as_str}' found in _options in NotebookPage")
