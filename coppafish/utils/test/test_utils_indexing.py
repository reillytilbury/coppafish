from coppafish.utils import indexing


def test_unique():
    assert indexing.unique([(0,), (1,), (2,)]) == [(0,), (1,), (2,)]
    assert indexing.unique([(0,), (0,), (2,)]) == [(0,), (2,)]
    assert indexing.unique([(0,), (1,), (2,)], 0) == [(0,), (1,), (2,)]
    assert indexing.unique([(0,), (0,), (1,)], 0) == [(0,), (1,)]
    assert indexing.unique([(1,), (1,), (1,)], 0) == [(1,)]
    assert indexing.unique([(0, 0), (1, 0), (2, 1)], 1) == [(0, 0), (2, 1)]
