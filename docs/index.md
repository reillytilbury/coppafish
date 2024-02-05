## Coppafish

Coppafish is an open source data analysis software for Combinatorial padlOck-Probe-Amplified Fluorescence In Situ 
Hybridization (coppafish) datasets. A series of 3D microscope images are arranged in tiles, rounds and channels. For 
each sequencing round, every gene being considered is fluoresced by a dye. Then, by the end of all rounds, each gene 
type has a unique sequence of dyes, called the gene code. Some vocabulary might be unfamiliar, please see the 
[glossary](glossary.md) as a reference.

## Installation

### Prerequisites

* At least 48GB of RAM (recommended for tile sizes `58x2048x2048`).
* Python version 3.9 or 3.10.
* [Git](https://git-scm.com/) (optional) is used to store more precise versioning, i.e. a hash, to detect any changes 
    in software version.

### Environment

We suggest installing your coppafish software within an environment. This can be a `venv` or a `conda` (recommended).

#### Conda

For `conda`, build an environment by doing:
```console
conda create -n coppafish python=3.10
conda activate coppafish
```

#### venv

```console
python -m venv /path/to/new/virtual/environment
```
then, in Linux and MacOS:
```console
source /path/to/new/virtual/environment/bin/activate
```
or windows:
```console
/path/to/new/virtual/environment/Scripts/activate.bat
```

### Install

Our latest coppafish release can be cloned locally
```console
git clone --depth 1 https://github.com/reillytilbury/coppafish
```

then install coppafish with our optimised jax code (Linux only)
```console
cd coppafish
python -m pip install -r requirements-optimised.txt
python -m pip install -e .[optimised]
```

for the non-optimised code (Windows support), instead do
```console
cd coppafish
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you do not wish to keep a local copy of coppafish (i.e. not interested in updating versions later) then remove the 
`-e` from the pip install. Then the source code can be deleted after installing.

## Running

Coppafish can be run with a config file. In the terminal
```console
python -m coppafish /path/to/config.ini
```

Or using a python script
```python
from coppafish import run_pipeline

run_pipeline("/path/to/config.ini")
```
