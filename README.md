
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
python setup.py develop
```


Examples
---------

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


Citing
---------

Please cite the following if you happen to use JARVIS-Tools for a publication.

https://www.nature.com/articles/s41524-020-00440-1

Choudhary, K. et al. The joint automated repository for various integrated simulations (JARVIS) for data-driven materials design. npj Computational Materials, 6(1), 1-13 (2020).
