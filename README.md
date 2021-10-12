[![name](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/knc6/jarvis-tools-notebooks/blob/master/jarvis-tools-notebooks/AtomVisionExample.ipynb)
# Atomvision
Atomvision:  A deep learning framework for atomistic image data

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
conda create --name version python=3.8
source activate version
```

Now, let's install the package:


```
git clone https://github.com/usnistgov/atomvision.git
cd atomvision
pip install torchvision
python setup.py develop
```


Examples
---------
This example shows how to classify 2D-lattice (5 classes) for 2D-materials STM/STEM images.

Datasets can be generated with STM/STEM sections of the `data` folder with ``generate_data.py`` script or pre-populated image datasets can be downloaded with 'download.py`. We create two folders ``train_folder``, ``test_folder`` with sub-folders ``0,1,2,3,4,...`` for individual classes and they contain images for these classes in way train-test splits have proportionate amount of images.
An example for using pre-trained densenet on STEM JARVIS-DFT 2D dataset is given below. Change ``train_folder`` and ``test_folder`` paths in order to use a different dataset.

```
python atomvision/scripts/train_classifiers.py --model_name densenet --train_folder atomvision/data/classification/stem_jv2d/train_folder --test_folder atomvision/data/classification/stem_jv2d/test_folder
```

Note: the repository is under development.


Citing
---------

Please cite the following if you happen to use JARVIS-Tools for a publication.

https://www.nature.com/articles/s41524-020-00440-1

Choudhary, K. et al. The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design. npj Computational Materials, 6(1), 1-13 (2020).
