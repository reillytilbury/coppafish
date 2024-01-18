import itertools
from typing import Tuple, Optional, Union, List, Any

from ..setup import NotebookPage


def create(
    nbp_basic: NotebookPage,
    include_rounds: bool = True,
    include_channels: bool = True,
    include_seq_rounds: bool = True,
    include_seq_channels: bool = True,
    include_anchor_round: bool = False,
    include_anchor_channel: bool = False,
    include_preseq_round: bool = False,
    include_dapi_seq: bool = False,
    include_dapi_anchor: bool = False,
    include_dapi_preseq: bool = False,
) -> Union[List[Tuple[int, int, int]], List[Tuple[int, int]], List[Tuple[int]]]:
    """
    Create tile, round and/or channel indices to loops through. Used throughout the coppafish pipeline. The defaults
    are set to return only sequencing rounds and channels. If something is set to be included which does not exist in 
    the notebook, e.g. a dapi channel or an anchor round, then it will not be included in the output.

    Args:
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        include_rounds (bool, optional): include round indices. Default: True.
        include_channels (bool, optional): include channel indices. Default: True.
        include_seq_rounds (bool, optional): include sequencing rounds. Default: True.
        include_seq_channels (bool, optional): include sequencing channels (gathered for the sequencing rounds and 
            presequence round, if chosen). Default: True.
        include_anchor_round (bool, optional): include anchor round. Default: False.
        include_anchor_channel (bool, optional): include the anchor channel, only for the anchor round. Default: False.
        include_preseq_round (bool, optional): include presequencing round. Default: False.
        include_dapi_seq (bool, optional): include dapi channel in sequencing rounds. Default: False.
        include_dapi_anchor (bool, optional): include dapi channel in anchor round. Default: False.
        include_dapi_preseq (bool, optional): include dapi channel in presequencing round. Default: False.

    Returns:
        list of tuple[int, int, int]] or list of tuple[int, int] or list of tuple[int] or list or tuple]: a list of
            tuples, each tuple containing a unique tile, round and/or channel index.

    Notes:
        - If `include_rounds` is false, then `include_channels` must also be false since the channel indices are 
            dependent on the round type, so this would not make sense to resolve.
    """
    if not include_rounds:
        assert not include_channels, "Unable to remove rounds and keep channels"
    
    seq_rounds = nbp_basic.use_rounds.copy()
    seq_channels = nbp_basic.use_channels.copy()
    all_tiles = sorted([t for t in nbp_basic.use_tiles.copy()])
    all_rounds = sorted([
        r
        for r in include_seq_rounds * seq_rounds
        + nbp_basic.use_anchor * include_anchor_round * [nbp_basic.anchor_round]
        + nbp_basic.use_preseq * include_preseq_round * [nbp_basic.pre_seq_round]
    ])
    all_channels = [c for c in seq_channels]
    if (include_dapi_anchor or include_dapi_preseq or include_dapi_seq) and nbp_basic.dapi_channel is not None:
        all_channels += [nbp_basic.dapi_channel]
    all_indices = []
    # The indexed channels will change depending on the round and the parameters
    for t, r in itertools.product(all_tiles, all_rounds):
        for c in all_channels:
            including = False
            if r in seq_rounds:
                if c in seq_channels and include_seq_channels:
                    including = True
                if c == nbp_basic.dapi_channel and include_dapi_seq:
                    including = True
            elif r == nbp_basic.anchor_round:
                if c == nbp_basic.dapi_channel and include_dapi_anchor:
                    including = True
                if c == nbp_basic.anchor_channel and include_anchor_channel:
                    including = True
            elif r == nbp_basic.pre_seq_round:
                if c in seq_channels and include_seq_channels:
                    including = True
                if c == nbp_basic.dapi_channel and include_dapi_preseq:
                    including = True
            if including:
                all_indices.append((t, r, c, ))
    output = []
    for t, r, c in all_indices:
        new_index = (t, )
        if include_rounds:
            new_index += (r, )
        if include_channels:
            new_index += (c, )
        output.append(new_index)
    # Remove any duplicate indices
    output = list(set(output))
    return sorted(output)


def unique(indices: List[Tuple[Any]], axis: Optional[int] = None) -> List[Tuple[Any]]:
    """
    Returns a list of indices that have a unique value in the `axis` index of the tuple. If a value in `axis` is seen 
    multiple times in indices, then the one that appears first is taken.

    Args:
        indices (list of tuple[any]): list of indices.
        axis (int, optional): axis in tuple to compare for uniqueness. Default: compute over all axes.

    Returns:
        list of tuple[any]: first unique indices along tuple axis number `axis`.
    """
    assert len(indices) != 0, "indices must not be empty"
    assert len(set([len(t) for t in indices])) == 1, "All tuples inside of indices must be the same length"
    if axis is not None:
        assert axis < len(indices[0]), "axis must be a dimension index in the tuples"

    unique_values = []
    unique_indices = []
    for index in indices:
        if axis is None:
            value = index
        else:
            value = index[axis]
        if value not in unique_values:
            unique_values.append(value)
            unique_indices.append(index)
    return unique_indices
