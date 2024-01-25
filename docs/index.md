## Installation

### Prerequisites

* A python version must be installed, see [homepage](https://github.com/reillytilbury/coppafish) for supported versions.
* [Git](https://git-scm.com/) (optional) is used to store more precise versioning, i.e. a hash, to detect any changes 
    in software version.

### Environment

We suggest installing your coppafish software within an environment. This can be a `venv` or a `conda` (recommended) 
environment.

#### Conda

For `conda`, build an environment by doing:
```console
conda create -n coppafish python=3.10
conda activate coppafish
```

#### Venv

in `venv`:
```console
python -m venv /path/to/new/virtual/environment
```
in Linux and MacOS:
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
pip install -r requirements-optimised.txt
pip install -e .[optimised]
```

for the non-optimised code (Windows support), instead do
```console
cd coppafish
pip install -r requirements.txt
pip install -e .
```

If you do not wish to keep a local copy of coppafish (i.e. not interested in updating versions later) then remove the 
`-e` from the pip install.
