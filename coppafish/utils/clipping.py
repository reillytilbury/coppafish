import numpy as np
import scipy


def contains_adjacent_max_pixels(image: np.ndarray) -> bool:
    """
    Computes whether or not the given image contains adjacent pixels at the greatest possible pixel value based on the
    image's integer datatype. Pixels are considered adjacent when one index shift can take you to the other pixel.

    Args:
        - image (`(Any shape) ndarray[int_]`): image to check.
    """
    assert type(image) is np.ndarray

    pixel_max = np.iinfo(image.dtype).max
    maximum_positions = np.array((image == pixel_max).nonzero()).T
    if maximum_positions.size == 0:
        return False
    tree = scipy.spatial.KDTree(maximum_positions)
    return len(tree.query_pairs(r=1)) > 0
