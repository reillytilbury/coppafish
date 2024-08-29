from typing import Optional, Tuple, Union

import numpy as np
import scipy

from .. import log
from ..setup import NotebookPage


class OutOfBoundsError(Exception):
    def __init__(self, var_name: str, oob_val: float, min_allowed: float, max_allowed: float):
        """
        Error raised because `oob_val` is outside expected range between
        `min_allowed` and `max_allowed` inclusive.

        Args:
            var_name: Name of variable testing.
            oob_val: Value in array that is not in expected range.
            min_allowed: Smallest allowed value i.e. `>= min_allowed`.
            max_allowed: Largest allowed value i.e. `<= max_allowed`.
        """
        self.message = (
            f"\n{var_name} contains the value {oob_val}."
            f"\nThis is outside the expected inclusive range between {min_allowed} and {max_allowed}"
        )
        super().__init__(self.message)


class NoFileError(Exception):
    def __init__(self, file_path: str):
        """
        Error raised because `file_path` does not exist.

        Args:
            file_path: Path to file of interest.
        """
        self.message = f"\nNo file with the following path:\n{file_path}\nexists"
        super().__init__(self.message)


class EmptyListError(Exception):
    def __init__(self, var_name: str):
        """
        Error raised because the variable indicated by `var_name` contains no data.

        Args:
            var_name: Name of list or numpy array
        """
        self.message = f"\n{var_name} contains no data"
        super().__init__(self.message)


def check_shape(array: np.ndarray, expected_shape: Union[list, tuple, np.ndarray]) -> bool:
    """
    Checks to see if `array` has the shape indicated by `expected_shape`.

    Args:
        array: Array to check the shape of.
        expected_shape: `int [n_array_dims]`.
            Expected shape of array.

    Returns:
        `True` if shape of array is correct.
    """
    correct_shape = array.ndim == len(expected_shape)  # first check if number of dimensions are correct
    if correct_shape:
        correct_shape = np.abs(np.array(array.shape) - np.array(expected_shape)).max() == 0
    return correct_shape


class ShapeError(Exception):
    def __init__(self, var_name: str, var_shape: tuple, expected_shape: tuple):
        """
        Error raised because variable indicated by `var_name` has wrong shape.

        Args:
            var_name: Name of numpy array.
            var_shape: Shape of numpy array.
            expected_shape: Expected shape of numpy array.
        """
        self.message = f"\nShape of {var_name} is {var_shape} but should be {expected_shape}"
        super().__init__(self.message)


class TiffError(Exception):
    def __init__(self, scale_tiff: float, scale_nbp: float, shift_tiff: int, shift_nbp: int):
        """
        Error raised because parameters used to produce tiff files are different to those in the current notebook.

        Args:
            scale_tiff: Scale factor applied to tiff. Found from tiff description.
            scale_nbp: Scale factor applied to tiff. Found from `nb.scale.scale`.
            shift_tiff: Shift applied to tiff to ensure pixel values positive. Found from tiff description.
            shift_nbp: Shift applied to tiff to ensure pixel values positive.
                Found from `nb.basic_info.tile_pixel_value_shift`.
        """
        self.message = (
            f"\nThere are differences between the parameters used to make the tiffs and the parameters "
            f"in the Notebook:"
        )
        if scale_tiff != scale_nbp:
            self.message = (
                self.message + f"\nScale used to make tiff was {scale_tiff}."
                f"\nCurrent scale in extract_params notebook page is {scale_nbp}."
            )
        if shift_tiff != shift_nbp:
            self.message = (
                self.message + f"\nShift used to make tiff was {shift_tiff}."
                f"\nCurrent tile_pixel_value_shift in basic_info notebook page is "
                f"{shift_nbp}."
            )
        super().__init__(self.message)


def check_color_nan(colors: np.ndarray, nbp_basic: NotebookPage) -> None:
    """
    `colors` should only contain the `invalid_value` in rounds/channels not in use_rounds/channels.
    This raises an error if this is not the case or if a round/channel not in use_rounds/channels
    contains a value other than `invalid_value`.
    `invalid_value = -nbp_basic.tile_pixel_value_shift` if colors is integer i.e. the non-normalised colors,
    usually spot_colours.
    `invalid_value = np.nan` if colors is float i.e. the normalised colors or most likely the bled_codes.

    Args:
        colors: `int or float [n_codes x n_rounds x n_channels]` \
            `colors[s, r, c]` is the color for code `s` in round `r`, channel `c`. \
            This is likely to be `spot_colours` if `int` or `bled_codes` if `float`.
        nbp_basic: basic_info NotebookPage. Requires values for `n_rounds`, `n_channels`, `use_rounds`, \
            `use_channels` and `tile_pixel_value_shift`.
    """
    # No idea what the hell is happening here but it works...
    diff_to_int = np.array([], dtype=int)
    not_nan = ~np.isnan(colors)
    diff_to_int = np.append(diff_to_int, [np.round(colors[not_nan]).astype(int) - colors[not_nan]])
    if np.abs(diff_to_int).max() == 0:
        # if not normalised, then invalid_value is an integer value that is impossible for a spot_color to be
        invalid_value = -nbp_basic.tile_pixel_value_shift
    else:
        # if is normalised then expect nan value to be normal np.nan.
        invalid_value = np.nan

    # decide which rounds/channels should be ignored i.e. only contain invalid_value.
    n_spots, n_rounds, n_channels = colors.shape
    if n_rounds == nbp_basic.n_rounds and n_channels == nbp_basic.n_channels:
        use_rounds = nbp_basic.use_rounds
        use_channels = nbp_basic.use_channels
    elif n_rounds == len(nbp_basic.use_rounds) and n_channels == len(nbp_basic.use_channels):
        use_rounds = np.arange(n_rounds)
        use_channels = np.arange(n_channels)
    else:
        log.error(ColorInvalidError(colors, nbp_basic, invalid_value))

    ignore_rounds = np.setdiff1d(np.arange(n_rounds), use_rounds)
    for r in ignore_rounds:
        unique_vals = np.unique(colors[:, r, :])
        for val in unique_vals:
            if np.isnan(invalid_value) and not np.isnan(val):
                log.error(ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r))
            if not np.isnan(invalid_value) and not invalid_value in unique_vals:
                log.error(ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r))
            if not np.isnan(invalid_value) and not np.array_equal(val, invalid_value, equal_nan=True):
                log.error(ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r))

    ignore_channels = np.setdiff1d(np.arange(n_channels), use_channels)
    for c in ignore_channels:
        unique_vals = np.unique(colors[:, :, c])
        for val in unique_vals:
            if np.isnan(invalid_value) and not np.isnan(val):
                log.error(ColorInvalidError(colors, nbp_basic, invalid_value, channel_no=c))
            if not np.isnan(invalid_value) and not invalid_value in unique_vals:
                log.error(ColorInvalidError(colors, nbp_basic, invalid_value, channel_no=c))
            if not np.isnan(invalid_value) and not np.array_equal(val, invalid_value, equal_nan=True):
                log.error(ColorInvalidError(colors, nbp_basic, invalid_value, channel_no=c))

    # see if any spots contain invalid_values.
    use_colors = colors[np.ix_(np.arange(n_spots), use_rounds, use_channels)]
    if np.array_equal(invalid_value, np.nan, equal_nan=True):
        nan_codes = np.where(np.isnan(use_colors))
    else:
        nan_codes = np.where(use_colors == invalid_value)
    n_nan_spots = nan_codes[0].size
    if n_nan_spots > 0:
        log.info(f"{n_nan_spots=}")
        s = nan_codes[0][0]
        # round, channel number in spot_colours different from in use_spot_colors.
        r = np.arange(n_rounds)[nan_codes[1][0]]
        c = np.arange(n_channels)[nan_codes[2][0]]
        log.error(ColorInvalidError(colors, nbp_basic, invalid_value, round_no=r, channel_no=c, code_no=s))


class ColorInvalidError(Exception):
    def __init__(
        self,
        colors: np.ndarray,
        nbp_basic: NotebookPage,
        invalid_value: float,
        round_no: Optional[int] = None,
        channel_no: Optional[int] = None,
        code_no: Optional[int] = None,
    ):
        """
        Error raised because `spot_colours` contains a `invalid_value` where it should not.

        Args:
            colors: `int or float [n_codes x n_rounds x n_channels]`
                `colors[s, r, c]` is the color for code `s` in round `r`, channel `c`.
                This is likely to be `spot_colours` if `int` or `bled_codes` if `float`.
            nbp_basic: basic_info NotebookPage
            invalid_value: This is the value that colors should only be in rounds/channels not used.
                Likely to be np.nan if colors is float or -nbp_basic.tile_pixel_value_shift if integer.
            round_no: round to flag error for.
            channel_no: channel to flag error for.
            code_no: Spot or gene index to flag error for.
        """
        n_spots, n_rounds, n_channels = colors.shape
        if round_no is not None and code_no is None:
            self.message = (
                f"colors contains a value other than invalid_value={invalid_value} in round {round_no}\n"
                f"which is not in use_rounds = {nbp_basic.use_rounds}."
            )
        elif channel_no is not None and code_no is None:
            self.message = (
                f"colors contains a value other than invalid_value={invalid_value} in channel {channel_no}\n"
                f"which is not in use_channels = {nbp_basic.use_channels}."
            )
        elif round_no is not None and channel_no is not None and code_no is not None:
            self.message = (
                f"colors contains a invalid_value={invalid_value} for code {code_no}, round {round_no}, "
                f"channel {channel_no}.\n"
                f"There should be no invalid_values in this round and channel."
            )
        else:
            self.message = (
                f"colors has n_rounds = {n_rounds} and n_channels = {n_channels}.\n"
                f"This is neither matches the total_rounds = {nbp_basic.n_rounds} and "
                f"total_channels = {nbp_basic.n_channels}\n"
                f"nor the number of use_rounds = {len(nbp_basic.use_rounds)} and use_channels = "
                f"{len(nbp_basic.use_channels)}"
            )
        super().__init__(self.message)


def compare_spots(
    spot_positions_0: np.ndarray,
    spot_gene_indices_0: np.ndarray,
    spot_positions_1: np.ndarray,
    spot_gene_indices_1: np.ndarray,
    distance_threshold: float,
) -> Tuple[np.ndarray[np.int8], int]:
    """
    Compare two collections of spots and assign each spot in the 0 collection either true positive, wrong positive,
    false positive. If a spot in collection 0 is matched to a spot in collection 1 if close together and of the same
    gene index, then this spot is a true positive. Any remaining spots are matched to close together spots in
    collection 1 also left unmatched and assigned as wrong positives. Remaining spots in collection 0 are false
    positives. If there are still unassigned spots in collection 1, these add to the false negative count. When
    multiple matches are found, the matched spots are the ones appearing first in the arrays for deterministic
    results. Collection 1 acts as the ground truth dataset.

    Args:
        - (`(n_spots_0 x 3) ndarray[float]`) spot_positions_0: spot collection 0 positions. spot_positions_0[i] is the
            ith spot's y, x, and z position.
        - (`(n_spots_0) ndarray[int]`) spot_gene_indices_0: spot gene indices.
        - (`(n_spots_1 x 3) ndarray[float]`) spot_positions_1: spot collection 1 positions.
        - (`(n_spots_1) ndarray[int]`) spot_gene_indices_1: spot collection 1 gene indices.
        - (float) distance_threshold: spot's are matched if their distance is no greater than distance_threshold.

    Returns:
        - spot_assignments (`(n_spots_0) ndarray[int8]): each collection 0 spot is given a label. 0 represents a true
            positive, 1 represents a wrong positive, 2 represents a false positive.
        - n_false_negatives (int): the number of false negative spots.
    """
    assert type(spot_positions_0) is np.ndarray
    assert spot_positions_0.ndim == 2
    assert spot_positions_0.shape[1] == 3
    assert type(spot_gene_indices_0) is np.ndarray
    assert spot_gene_indices_0.ndim == 1
    spot_gene_indices_0.shape[0] == spot_positions_0.shape[0]
    assert type(spot_positions_1) is np.ndarray
    assert spot_positions_1.ndim == 2
    assert spot_positions_1.shape[1] == 3
    assert type(spot_gene_indices_1) is np.ndarray
    assert spot_gene_indices_1.ndim == 1
    spot_gene_indices_1.shape[0] == spot_positions_1.shape[0]
    assert type(distance_threshold) is float
    assert distance_threshold > 0

    spot_assignments = np.zeros_like(spot_gene_indices_0, np.int8) - 1
    n_spots_0 = spot_positions_0.shape[0]
    n_spots_1 = spot_positions_1.shape[0]
    unmatched_spot_0 = np.ones(n_spots_0, bool)
    unmatched_spot_1 = np.ones(n_spots_1, bool)

    # Loop through spots, find matches of the same gene index.
    for i in range(n_spots_0):
        i_gene = spot_gene_indices_0[i]
        if unmatched_spot_1.sum() == 0:
            break
        spot_positions_1_unmatched = np.full_like(spot_positions_1, -999_999, np.float32)
        spot_positions_1_unmatched[unmatched_spot_1] = spot_positions_1[unmatched_spot_1]
        kdtree = scipy.spatial.KDTree(spot_positions_1_unmatched)
        matching_spots_1 = kdtree.query_ball_point(spot_positions_0[i], r=distance_threshold, workers=-1)
        if len(matching_spots_1) == 0:
            continue
        is_matching_gene_1 = np.zeros_like(spot_gene_indices_1, bool)
        is_matching_gene_1[matching_spots_1] = spot_gene_indices_1[matching_spots_1] == i_gene
        if is_matching_gene_1.sum() == 0:
            continue
        first_matching_spot_1 = is_matching_gene_1.nonzero()[0][0]
        unmatched_spot_0[i] = False
        unmatched_spot_1[first_matching_spot_1] = False
        spot_assignments[i] = 0

    # Now loop through unmatched 0 spots again. Any matching spots with wrong gene are now considered wrong positives.
    for i in unmatched_spot_0.nonzero()[0]:
        i_gene = spot_gene_indices_0[i]
        spot_positions_1_unmatched = np.full_like(spot_positions_1, -999_999, np.float32)
        spot_positions_1_unmatched[unmatched_spot_1] = spot_positions_1[unmatched_spot_1]
        kdtree = scipy.spatial.KDTree(spot_positions_1_unmatched)
        matching_spots_1 = kdtree.query_ball_point(spot_positions_0[i], r=distance_threshold, workers=-1)
        if len(matching_spots_1) == 0:
            spot_assignments[i] = 2
            continue
        is_matching_gene_1 = spot_gene_indices_1[matching_spots_1] == i_gene
        assert is_matching_gene_1.sum() == 0
        first_matching_spot_1 = matching_spots_1[0]
        unmatched_spot_0[i] = False
        unmatched_spot_1[first_matching_spot_1] = False
        spot_assignments[i] = 1

    false_negative_count = unmatched_spot_1.sum()

    return spot_assignments, false_negative_count
