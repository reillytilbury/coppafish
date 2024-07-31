from setuptools import setup, find_packages

with open("coppafish/_version.py", "r") as f:
    exec(f.read())

with open("README.md", "r") as f:
    long_desc = f.read()

packages = [folder for folder in find_packages() if folder[-5:] != ".test"]  # Get rid of test packages

setup(
    name="coppafish",
    version=__version__,
    description="coppaFISH software for Python",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    author="Josh Duffield",
    author_email="jduffield65@gmail.com",
    maintainer="Reilly Tilbury, Paul Shuker",
    maintainer_email="reillytilbury@gmail.com, paul.shuker@outlook.com",
    license="MIT",
    python_requires=">3.8, <3.11",
    url="https://reillytilbury.github.io/coppafish/",
    packages=packages,
    install_requires=[
        "dask",
        "distinctipy",
        "h5py",
        "magicgui",
        "matplotlib",
        "mplcursors",
        "napari",
        "nd2",
        "numpy<2.0.0",
        "numpy_indexed",
        "opencv-python-headless",
        "pandas",
        "psutil",
        "PyQt5",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "torch",
        "tqdm",
        "zarr",
    ],
    package_data={
        "coppafish.setup": [
            "settings.default.ini",
            "notebook_comments.json",
            "dye_camera_laser_raw_intensity.csv",
            "default_bleed.npy",
            "default_psf.npz",
        ],
        "coppafish.plot.results_viewer": ["cell_color.csv", "cellClassColors.json", "gene_color.csv"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: Unix",
    ],
)
