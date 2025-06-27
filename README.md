
[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/AtomVisionExample.ipynb)
![alt text](https://github.com/usnistgov/atomvision/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/usnistgov/atomvision/branch/master/graph/badge.svg?token=S5X4OYC80V)](https://codecov.io/gh/usnistgov/atomvision)
[![PyPI version](https://badge.fury.io/py/atomvision.svg)](https://badge.fury.io/py/atomvision)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/usnistgov/atomvision)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/usnistgov/atomvision)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/usnistgov/atomvision)
[![Downloads](https://pepy.tech/badge/atomvision)](https://pepy.tech/project/atomvision)
# Atomvision

⚠️ This repository is no longer maintained.
For the latest updates and continued development, please visit: https://github.com/atomgptlab/atomvision

# Table of Contents
* [Introduction](#intro)
* [Installation](#install)
* [Examples](#example)
* [Reference](#reference)
* [How to contribute](#contrib)
* [Correspondence](#corres)
* [Funding support](#fund)


<a name="intro"></a>
Introduction
-------------------------
Atomvision is a deep learning framework for atomistic image data.



<p align="center">
   <img src="https://github.com/usnistgov/atomvision/blob/develop/atomvision/AVFigure.PNG" alt="AtomVision"  width="600"/>
</p>

<a name="install"></a>
Installation
-------------------------
First create a conda environment:
Install miniconda environment from https://conda.io/miniconda.html
Based on your system requirements, you'll get a file something like 'Miniconda3-latest-XYZ'.

Now,

```
bash Miniconda3-latest-Linux-x86_64.sh (for linux)
bash Miniconda3-latest-MacOSX-x86_64.sh (for Mac)
```
Download 32/64 bit python 3.6 miniconda exe and install (for windows)
Now, let's make a conda environment, say "version", choose other name as you like::
```
conda create --name vision python=3.8
source activate vision
```

Now, let's install the package:

#### Method 1 (using setup.py):

```
git clone https://github.com/usnistgov/atomvision.git
cd atomvision
python setup.py develop
```

#### Method 2 (using pypi):

As an alternate method, AtomVision can also be installed using `pip` command as follows:
```
pip install atomvision
```

<a name="example"></a>
Examples
---------


#### Generating STEM image with convolution approximation: graphene example

```
stem_conv.py --file_path atomvision/tests/POSCAR --output_path STEM.png
```


#### 2D-Bravais lattice classification example
This example shows how to classify 2D-lattice (5 Bravais classes) for 2D-materials STM/STEM images.

We will use images``sample_data`` folder. It was generated with ``generate_stem.py`` script. There are  two folders ``train_folder``, ``test_folder`` with sub-folders ``0,1,2,3,4,...`` for individual classes and they contain images for these classes.

```
train_classifier_cnn.py --model densenet --train_folder atomvision/sample_data/test_folder --test_folder atomvision/sample_data/test_folder --epochs 5 --batch_size 16
```


#### Generating a t-SNE  plot

```
train_tsne.py --data_dir atomvision/sample_data/test_folder
```

#### Generative Adversarial Network

```
train_gan.py --dataset_path atomvision/sample_data/test_folder/0 --epochs 2
```

#### Autoencoder

```
train_autoencoder.py --train_folder atomvision/sample_data/test_folder --test_folder atomvision/sample_data/test_folder --epochs 10
```


<a name="reference"></a>
Reference
---------

1) [AtomVision: A machine vision library for atomistic images](https://arxiv.org/abs/2212.02586)

2) [The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design](https://www.nature.com/articles/s41524-020-00440-1)

3) [Computational scanning tunneling microscope image database](https://www.nature.com/articles/s41597-021-00824-y)

Please see detailed publications list [here](https://jarvis-tools.readthedocs.io/en/master/publications.html).

<a name="contrib"></a>
How to contribute
-----------------

For detailed instructions, please see [Contribution instructions](https://github.com/usnistgov/jarvis/blob/master/Contribution.rst)

<a name="corres"></a>
Correspondence
--------------------

Please report bugs as Github issues (https://github.com/usnistgov/atomvision/issues) or email to kamal.choudhary@nist.gov.

<a name="fund"></a>
Funding support
--------------------

NIST-MGI (https://www.nist.gov/mgi).

Code of conduct
--------------------

Please see [Code of conduct](https://github.com/usnistgov/jarvis/blob/master/CODE_OF_CONDUCT.md)

