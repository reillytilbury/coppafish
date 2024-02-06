import time
import pytest



def square(x):
    return x * x


def add(args):
    return args[0] + args[1]


@pytest.mark.pytorch
def test_multiprocess_function():
    from coppafish.utils import multiprocess_pytorch
    assert multiprocess_pytorch.multiprocess_function(square, [0, 500, 2]) == [0, 500*500, 4]
    assert multiprocess_pytorch.multiprocess_function(add, [(1, 2), (5, 100), (0, 1)]) == [3, 105, 1]


if __name__ == "__main__":
    test_multiprocess_function()
