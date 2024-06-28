import os
from pathlib import PurePath
import shutil
import tempfile

import numpy as np
import zarr

from coppafish import utils
from coppafish.setup.notebook import Notebook
from coppafish.setup.notebook_page import NotebookPage


def test_notebook_creation() -> None:
    rng = np.random.RandomState(0)

    nb_path = os.path.join(os.getcwd(), ".notebook_test")
    if os.path.isdir(nb_path):
        shutil.rmtree(nb_path)
    config_path = os.path.abspath("dslkhgdsjlgh")
    nb = Notebook(nb_path, config_path)

    assert nb.has_page("debug") == False
    assert nb.config_path == config_path

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

    def _check_variables(nb: Notebook):
        assert nb.has_page("debug")
        assert np.allclose(nb.debug.a, a)
        assert np.allclose(nb.debug.b, b)
        assert np.allclose(nb.debug.c, c)
        assert np.allclose(nb.debug.d, d)
        assert type(nb.debug.d) is list
        assert nb.debug.e == utils.base.deep_convert(e, list)
        assert type(nb.debug.e) is list
        assert np.allclose(nb.debug.f, f)
        assert nb.debug.g is g
        assert np.allclose(nb.debug.h, h)
        assert nb.debug.i == i
        assert np.allclose(nb.debug.j, j)
        assert np.allclose(nb.debug.k, k)
        assert np.allclose(nb.debug.l, l)
        assert (nb.debug.m == m).all()
        assert np.allclose(nb.debug.n, n)
        zarray_path = os.path.abspath(nb.debug.o.store.path)
        assert os.path.isdir(zarray_path)
        assert len(os.listdir(zarray_path)) > 0
        assert PurePath(nb_path) in PurePath(zarray_path).parents
        zgroup_path = os.path.abspath(nb.debug.p.store.path)
        assert os.path.isdir(zgroup_path)
        assert len(os.listdir(zgroup_path)) > 0
        assert type(nb.debug.p["subgroup"]) is zarr.Group
        assert type(nb.debug.p["subarray.zarr"]) is zarr.Array
        assert nb.debug.p["subarray.zarr"].shape == (10, 5)

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
        nb_page.d = (5, "4")
        nb_page.d = (5, 0.5)
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

    try:
        nb += nb_page
        assert False, f"Should crash when adding an unfinished notebook page"
    except ValueError:
        pass

    temp_zarr = tempfile.TemporaryDirectory()
    array_saved = np.zeros((4, 8), dtype=np.float32)
    zarr_array_temp = zarr.open_array(
        store=temp_zarr.name, shape=array_saved.shape, dtype="|f4", zarr_version=2, chunks=(2, 4), mode="w"
    )
    zarr_array_temp[:] = array_saved.copy()

    assert nb_page.get_unset_variables() == ("o", "p")

    nb_page.o = zarr_array_temp
    del zarr_array_temp

    assert len(nb_page.get_unset_variables()) == 1
    assert nb_page.name == "debug"

    try:
        nb += nb_page
        assert False, f"Shoul not be able to add an unfinished page to the notebook"
    except ValueError:
        pass

    temp_zgroup = tempfile.TemporaryDirectory()
    group = zarr.group(store=temp_zgroup.name, zarr_version=2)
    group.create_dataset("subarray.zarr", shape=(10, 5), dtype=np.int16)
    group.create_group("subgroup")
    nb_page.p = group

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

    nb > "debug"
    nb_page > "o"

    _check_variables(nb)
    del nb_page
    nb.resave()
    _check_variables(nb)
    del nb.debug.a
    a = 10
    nb.debug.a = a
    nb.resave()
    _check_variables(nb)

    del nb
    nb = Notebook(nb_path)
    _check_variables(nb)

    # Check that the resave function can safely remove pages.
    del nb.debug
    nb.resave()
    assert not nb.has_page("debug")
    assert not os.path.exists(os.path.join(nb_path, "debug"))

    nb = Notebook(nb_path)
    assert not nb.has_page("debug")
    assert not os.path.exists(os.path.join(nb_path, "debug"))

    # Delete the temporary notebook once done testing.
    shutil.rmtree(nb_path)

    # Clean any temporary files/directories.
    temp_zarr.cleanup()
    temp_zgroup.cleanup()
