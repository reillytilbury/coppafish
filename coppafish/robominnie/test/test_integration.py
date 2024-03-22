import os
import numpy as np
import warnings
import pytest

from coppafish import Viewer, BuildPDF, Notebook
from coppafish.robominnie.robominnie import RoboMinnie
from coppafish.plot.register.diagnostics import RegistrationViewer


def get_robominnie_scores(rm: RoboMinnie, include_omp: bool = True) -> None:
    print(rm.compare_spots("ref"))
    overall_score = rm.overall_score()
    print(f"Overall score: {round(overall_score*100, 1)}%")
    if overall_score < 0.75:
        warnings.warn(UserWarning("Integration test passed, but the overall reference spots score is < 75%"))

    if not include_omp:
        return
    print(rm.compare_spots("omp"))
    overall_score = rm.overall_score()
    print(f"Overall score: {round(overall_score*100, 1)}%")
    if overall_score < 0.75:
        warnings.warn(UserWarning("Integration test passed, but the overall OMP spots score is < 75%"))
    del rm


@pytest.mark.integration
@pytest.mark.slow
def test_integration_smallest() -> Notebook:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, sequencing rounds, one `5x100x100` tile.

    Returns:
        Notebook: complete coppafish Notebook.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(n_planes=5, n_tile_yx=(150, 150), include_presequence=False, include_dapi=False)
    robominnie.generate_gene_codes(4)
    robominnie.generate_pink_noise()
    robominnie.add_spots(5000)
    robominnie.save_raw_images(output_dir)
    nb = robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie
    return nb


@pytest.mark.integration
def test_integration_small_two_tile():
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, sequencing rounds, one `4x100x100` tile.

    Returns:
        Notebook: complete coppafish Notebook.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(n_channels=4, n_planes=5, n_tile_yx=(100, 100), n_tiles_y=2)
    robominnie.generate_gene_codes(4)
    robominnie.generate_pink_noise()
    robominnie.add_spots(500)
    robominnie.save_raw_images(output_dir)
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.integration
@pytest.mark.slow
def test_integration_002() -> None:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, DAPI image, presequence round, sequencing rounds, one tile.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie()
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(spot_size_pixels_dapi=np.array([9, 9, 9]), include_dapi=True, spot_amplitude_dapi=0.05)
    robominnie.save_raw_images(output_dir=output_dir)
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.integration
@pytest.mark.slow
def test_integration_non_symmetric(include_stitch: bool = True, include_omp: bool = True) -> Notebook:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, DAPI channels, presequencing round, and sequencing rounds, `2` connected 11x75x75 tiles,
    aligned along the x axis. There are 7 sequencing rounds and 9 channels/dyes so the bleed matrix is non-symmetric.

    Args:
        include_stitch (bool, optional): run stitch. Default: true.
        include_omp (bool, optional): run OMP. Default: true.

    Returns:
        Notebook: final notebook.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie(n_planes=4, n_tile_yx=(141, 141), n_channels=9, n_tiles_x=2)
    robominnie.generate_gene_codes(10)
    robominnie.generate_pink_noise()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(include_dapi=True, spot_size_pixels_dapi=np.array([9, 9, 9]), spot_amplitude_dapi=0.05)
    robominnie.save_raw_images(output_dir=output_dir)
    nb = robominnie.run_coppafish(include_stitch=include_stitch, include_omp=include_omp)
    if not include_omp or not include_stitch:
        return nb
    get_robominnie_scores(robominnie)
    del robominnie
    return nb


@pytest.mark.integration
@pytest.mark.slow
def test_integration_004() -> None:
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, DAPI image, presequence round, sequencing rounds, one tile. No DAPI channel registration.
    """
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = RoboMinnie()
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    # Add spots to DAPI image as larger spots
    robominnie.add_spots(15000, spot_size_pixels_dapi=np.array([9, 9, 9]), include_dapi=True, spot_amplitude_dapi=0.05)
    robominnie.save_raw_images(output_dir=output_dir, register_with_dapi=False)
    robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.integration
@pytest.mark.slow
def test_bg_subtraction() -> None:
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    rng = np.random.RandomState(0)

    robominnie = RoboMinnie(brightness_scale_factor=rng.rand(1, 9, 8) / 4 + 0.75)
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    robominnie.add_spots(
        gene_efficiency=0.5 * (rng.rand(20, 8) + 1),
        background_offset=1e-7 * rng.rand(15_000, 7),
        include_dapi=True,
        spot_size_pixels_dapi=np.asarray([9, 9, 9]),
        spot_amplitude_dapi=0.05,
    )
    robominnie.save_raw_images(output_dir=output_dir, register_with_dapi=False)
    nb = robominnie.run_coppafish()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.integration
@pytest.mark.slow
def test_viewers() -> None:
    """
    Make sure the coppafish plotting is working without crashing.

    Notes:
        - Requires a robominnie instance to have successfully run through first.
    """
    notebook_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), ".integration_dir/output_coppafish/notebook.npz"
    )
    if not os.path.isfile(notebook_path):
        return
    gene_colours_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir/gene_colours.csv")
    notebook = Notebook(notebook_path)
    Viewer(notebook, gene_marker_file=gene_colours_path)
    RegistrationViewer(notebook)


@pytest.mark.integration
@pytest.mark.slow
def test_pdf_builder() -> None:
    """
    Makes sure the BuildPDF class is working without crashing.

    Notes:
        - Requires a robominnie instance to have run through first to retrieve the notebook file.
    """
    notebook_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), ".integration_dir/output_coppafish/notebook.npz"
    )
    for file_name in os.listdir(os.path.dirname(notebook_path)):
        if file_name[-4:].lower() == ".pdf":
            os.remove(os.path.join(os.path.dirname(notebook_path), file_name))
    BuildPDF(notebook_path, auto_open=False)


if __name__ == "__main__":
    test_integration_small_two_tile()
    test_pdf_builder()
    test_viewers()
