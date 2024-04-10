import os

from coppafish import Viewer2D


def test_Viewer2D() -> None:
    #! Requires robominnie to have successfully run through at least up to call spots.
    print(os.path.dirname(os.path.realpath(__file__)))
    notebook_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
        "robominnie",
        "test",
        ".integration_dir",
        "output_coppafish",
        "notebook.npz",
    )
    assert os.path.isfile(notebook_path), "Failed to find notebook at\n" + notebook_path
    Viewer2D(notebook_path)


if __name__ == "__main__":
    test_Viewer2D()
