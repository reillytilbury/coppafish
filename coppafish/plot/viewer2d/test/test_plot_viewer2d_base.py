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
    gene_colours_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
        "robominnie",
        "test",
        ".integration_dir",
        "gene_colours.csv",
    )
    assert os.path.isfile(notebook_path), "Failed to find notebook at\n" + notebook_path
    assert os.path.isfile(gene_colours_path), "Failed to find gene markers at\n" + gene_colours_path
    Viewer2D(notebook_path, gene_marker_file=gene_colours_path)


if __name__ == "__main__":
    test_Viewer2D()
