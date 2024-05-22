import os
import zarr
import shutil

import numpy as np

from coppafish.setup.notebook_new import Notebook
from coppafish.setup.notebook_page_new import NotebookPage


def test_notebook_creation() -> None:
    nb_path = os.path.join(os.getcwd(), ".notebook_test")
    if os.path.isdir(nb_path):
        shutil.rmtree(nb_path)
    nb = Notebook(nb_path)

    assert nb.has_page("debug") == False

    nb_page: NotebookPage = NotebookPage("debug")
    nb_page.a = 5
    try:
        nb_page.b = 5
        assert False, "Should not be able to set a float type to an int"
    except TypeError:
        pass
    nb_page.b = 5.0
    nb_page.c = True
    try:
        nb_page.d = (5, "4", True)
        assert False, "Should not be able to set a tuple[int] type like this"
    except TypeError:
        pass
    nb_page.d = (4, 5, 6, 7)
    del nb_page.d
    nb_page.d = tuple()
    nb_page.e = ((0.4, 5.0), (2.0, 1.0, 4.5), tuple())
    nb_page.f = 3
    del nb_page.f
    nb_page.f = 3.0
    nb_page.g = None
    nb_page.h = None
    del nb_page.h
    nb_page.h = 4.3
    nb_page.i = "Hello, World"
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
    nb_page.j = np.zeros(10, dtype=np.float16)
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
    nb_page.k = np.zeros(10, dtype=np.int64)
    nb_page.l = np.zeros(10, dtype=bool)

    nb_page.m = np.zeros(5, dtype=str)
    nb_page.n = np.zeros(0, dtype=np.uint32)

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

    assert nb.has_page("debug")
    assert os.path.isdir(nb.debug.o)
    assert len(os.listdir(nb.debug.o)) > 0
    assert nb.debug.j.size == 10
    assert nb.debug.k.size == 10
    assert nb.debug.l.size == 10


if __name__ == "__main__":
    test_notebook_creation()
