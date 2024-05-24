import os
from pathlib import PurePath
import shutil

import numpy as np
import zarr

from coppafish.setup.notebook import Notebook
from coppafish.setup.notebook_page import NotebookPage


def test_notebook_creation() -> None:
    rng = np.random.RandomState(0)

    nb_path = os.path.join(os.getcwd(), ".notebook_test")
    if os.path.isdir(nb_path):
        shutil.rmtree(nb_path)
    nb = Notebook(nb_path)

    nb.config_path = "blahalbshkglvsf"
    try:
        nb.config_path = "djfhdersd"
        assert False, "Should not be allowed to set the config path twice"
    except ValueError:
        pass

    assert nb.has_page("debug") == False

    nb_page: NotebookPage = NotebookPage("debug")

    a = 5
    b = 5.0
    c = True
    d = (4, 5, 6, 7)
    e = ((0.4, 5.0), (2.0, 1.0, 4.5), tuple())
    f = 3.0
    g = None
    h = 4.3
    i = "Hello, World"
    j = rng.rand(3, 10).astype(dtype=np.float16)
    k = rng.randint(2000, size=10, dtype=np.int64)
    l = rng.randint(2, size=(3, 4, 6), dtype=bool)
    m = np.zeros(3, dtype=str)
    m[0] = "blah"
    n = rng.randint(200, size=(7, 8), dtype=np.uint32)

    nb_page.a = a
    try:
        nb_page.b = 5
        assert False, "Should not be able to set a float type to an int"
    except TypeError:
        pass
    nb_page.b = b
    nb_page.c = c
    try:
        nb_page.d = (5, "4", True)
        assert False, "Should not be able to set a tuple[int] type like this"
    except TypeError:
        pass
    nb_page.d = tuple()
    del nb_page.d
    nb_page.d = d
    nb_page.e = e
    nb_page.f = 3
    del nb_page.f
    nb_page.f = f
    nb_page.g = g
    nb_page.h = None
    del nb_page.h
    nb_page.h = h
    nb_page.i = i
    try:
        nb_page.j = np.zeros(10, dtype=int)
        assert False, "Should not be able to set a ndarray[float] type like this"
    except TypeError:
        pass
    try:
        nb_page.j = np.zeros(10, dtype=bool)
        assert False, "Should not be able to set a ndarray[float] type like this"
    except TypeError:
        pass
    nb_page.j = j
    try:
        nb_page.k = np.zeros(10, dtype=np.float32)
        assert False, "Should not be able to set a ndarray[int] type like this"
    except TypeError:
        pass
    try:
        nb_page.k = np.zeros(10, dtype=bool)
        assert False, "Should not be able to set a ndarray[int] type like this"
    except TypeError:
        pass
    nb_page.k = k
    nb_page.l = l

    nb_page.m = m
    nb_page.n = n

    zarr_path = os.path.join(os.getcwd(), ".test_array.zarr")
    array_saved = np.zeros((4, 8), dtype=np.float32)
    zarr_array_temp = zarr.open_array(
        store=zarr_path, shape=array_saved.shape, dtype="|f4", zarr_version=2, chunks=(2, 4), mode="w"
    )
    zarr_array_temp[:] = array_saved.copy()
    del zarr_array_temp

    assert nb_page.get_unset_variables() == ("o",)

    nb_page.o = zarr_path

    assert len(nb_page.get_unset_variables()) == 0
    assert nb_page.name == "debug"

    nb += nb_page

    try:
        nb.fake_variable = 4
        assert False, f"Should not be able to add integer variables to the notebook"
    except TypeError:
        pass
    try:
        nb += nb_page
        assert False, f"Should not be able to add the same page twice"
    except ValueError:
        pass

    def check_variables():
        assert nb.has_page("debug")
        assert np.allclose(nb.debug.a, a)
        assert np.allclose(nb.debug.b, b)
        assert np.allclose(nb.debug.c, c)
        assert np.allclose(nb.debug.d, d)
        assert type(nb.debug.d) is tuple
        assert nb.debug.e == e
        assert type(nb.debug.e) is tuple
        assert np.allclose(nb.debug.f, f)
        assert nb.debug.g is g
        assert np.allclose(nb.debug.h, h)
        assert nb.debug.i == i
        assert np.allclose(nb.debug.j, j)
        assert np.allclose(nb.debug.k, k)
        assert np.allclose(nb.debug.l, l)
        assert (nb.debug.m == m).all()
        assert np.allclose(nb.debug.n, n)
        assert os.path.isdir(nb.debug.o)
        assert len(os.listdir(nb.debug.o)) > 0
        assert PurePath(nb_path) in PurePath(nb.debug.o).parents

    nb > "debug"
    nb_page > "o"

    check_variables()
    del nb_page
    nb.resave()
    check_variables()
    del nb.debug.a
    a = 10
    nb.debug.a = a
    nb.resave()
    check_variables()

    del nb
    nb = Notebook(nb_path)
    check_variables()


if __name__ == "__main__":
    test_notebook_creation()
