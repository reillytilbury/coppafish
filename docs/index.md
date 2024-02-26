## Coppafish

Coppafish is an open source data analysis software for COmbinatorial Padlock-Probe-Amplified Fluorescence In Situ 
Hybridization (coppafish) datasets. A series of 3D microscope images are arranged in tiles, rounds and channels. For 
each sequencing round, every considered gene is fluoresced by a dye. By the end of all rounds, each gene has a unique 
barcode sequence of dyes, called the gene code. Some vocabulary might be unfamiliar, please see the 
[glossary](glossary.md) for reference.

## Installation

### Prerequisites

* At least 48GB of RAM for tile sizes `58x2048x2048`.
* Python version 3.9 or 3.10.
* [Git](https://git-scm.com/) (optional) is used to store more precise versioning, i.e. a hash, to detect any changes 
    in software version.

### Environment

Install coppafish software from within an environment. This can be a `venv` or `conda` (recommended) environment.

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

to install the optimised, [pytorch](https://github.com/pytorch) GPU code (Windows and Linux support)
```console
cd coppafish
python -m pip install --upgrade pip
python -m pip install -r requirements-pytorchgpu.txt
python -m pip install -e .
```

or for the optimised, [pytorch](https://github.com/pytorch) CPU code (Windows and Linux support)
```console
cd coppafish
python -m pip install --upgrade pip
python -m pip install -r requirements-pytorch.txt
python -m pip install -e .
```

or for the optimised [jax](https://github.com/google/jax) code (Linux only and is less stable)
```console
cd coppafish
python -m pip install --upgrade pip
python -m pip install -r requirements-optimised.txt
python -m pip install -e .
```

or for the slower, numpy-only reliant code (Windows and Linux support)
```console
cd coppafish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

this can be useful for people with limited disk space for large packages like jax and pytorch who do not mind 
sacrificing computation speed.

If you do not wish to keep a local copy of coppafish (i.e. not interested in `git pull`ing higher coppafish versions 
later) then remove the `-e` option. The source code can then be deleted after installing.

If pytorch GPU is installed, but no cuda device is found available, then coppafish will automatically revert back to 
CPU.
