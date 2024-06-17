import numpy as np

from coppafish.utils import clipping


def test_contains_adjacent_max_pixels() -> None:
    rng = np.random.RandomState(0)
    image = rng.randint(0, 100, (1, 3, 5, 11), np.int16)
    assert not clipping.contains_adjacent_max_pixels(image)
    image[0, 1, 0, 0] = np.iinfo(np.int16).max
    assert not clipping.contains_adjacent_max_pixels(image)
    image[0, 2, 0, 0] = np.iinfo(np.int16).max
    assert clipping.contains_adjacent_max_pixels(image)

    image = rng.randint(0, 100, (1, 3, 5, 11), np.int32)
    assert not clipping.contains_adjacent_max_pixels(image)
    image[0, 1, 0, 0] = np.iinfo(np.int32).max
    assert not clipping.contains_adjacent_max_pixels(image)
    image[0, 2, 0, 0] = np.iinfo(np.int32).max
    assert clipping.contains_adjacent_max_pixels(image)
    image[0, 2, 0, 0] = np.iinfo(np.int32).max
    image[0, 1, 4, 4] = np.iinfo(np.int32).max
    assert clipping.contains_adjacent_max_pixels(image)
