import os
import pytest

from coppafish import BuildPDF


@pytest.mark.slow
def test_BuildPDF():
    robominnie_notebook = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../robominnie/test/.integration_dir/output_coppafish/notebook.npz",
    )
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "unit_test_dir/diagnostics.pdf")
    if not os.path.isdir(os.path.dirname(output_path)):
        os.mkdir(output_path)
    if not os.path.isfile(robominnie_notebook):
        assert False, f"Could not find robominnie notebook at\n\t{robominnie_notebook}.\nRun an integration test first"

    BuildPDF(robominnie_notebook, output_dir=output_path, auto_open=False)
