## Coppafish

Coppafish is an open source data analysis software for COmbinatorial Padlock-Probe-Amplified Fluorescence In Situ
Hybridization (coppafish) datasets. A series of 3D microscope images are arranged into tiles, rounds and channels. For
each sequencing round, every considered gene is fluoresced by a dye. By the end of all rounds, each gene has a unique,
barcode-like sequence of dyes, called the gene code. For more details about coppafish's methodology, see the
[method](method.md). See [installation](#installation) on how to install our software, and [usage](basic_usage.md) to
run coppafish on your dataset. Some vocabulary might be unfamiliar, please see the [glossary](glossary.md) for
reference.

<figure markdown="span">
  ![Image title](images/coppafish_simple_schematic.png){ width="400" }
  <figcaption>Gene calling on a tile.</figcaption>
</figure>

## Installation

### Prerequisites

* Windows or Linux operating system. MacOS is not tested.
* At least 48GB of RAM for tile sizes `58x2048x2048`.
* Python version 3.9 or 3.10.
* [Git](https://git-scm.com/).

### Environment

Install coppafish software from within an environment. This can be a `venv` or `conda` (recommended) environment.

#### Conda

For `conda`, build an environment by doing:
``` bash
conda create -n coppafish python=3.10
conda activate coppafish
```

### Install

Our latest coppafish release can be cloned locally
``` bash
git clone --depth 1 https://github.com/reillytilbury/coppafish
```

install package dependencies and coppafish by
``` bash
cd coppafish
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install .
```

## Updating

Coppafish will not automatically install updates, but you will see a warning at the start of a pipeline if a new online
version is available.

If you already have the source code downloaded, navigate inside of the `coppafish` directory, then `#!bash git pull` 
the latest code changes. Then, follow the [install](#install) steps again, excluding the `#!bash git clone` command, 
while inside your coppafish environment.

If you do not have the source code downloaded anymore, follow all the [install](#install) again while inside your
coppafish environment.

You can verify your install by running `#!bash pip show coppafish` in the coppafish environment to check you have the 
latest version.
