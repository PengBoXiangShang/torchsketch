# TorchSketch
[![PyPI](https://img.shields.io/pypi/v/torchsketch)](https://pypi.org/project/torchsketch/) ![](https://img.shields.io/badge/language-Python-{green}.svg) ![](https://img.shields.io/npm/l/express.svg)

<div align=center><img src="https://github.com/PengBoXiangShang/torchsketch/blob/master/torchsketch/docs/others/torchsketch.gif"/></div>

TorchSketch is an open source software library for free-hand sketch oriented deep learning research, which is built on the top of [PyTorch](https://pytorch.org/).

**The project is under continuous update!**



## 1. Installation
TorchSketch is developed based on Python 3.7.

To avoid any conflicts with your existing Python setup, it's better to install TorchSketch into a standalone environment, e.g., an Anaconda virtual environment.

Assume that you have installed Anaconda. Please create a virtual environment before installation of TorchSketch, as follows.
```bash
# Create a virtual environment in Anaconda.
conda create --name ${CUSTOMIZED_ENVIRONMENT_NAME} python=3.7

# Activate it.
conda activate ${CUSTOMIZED_ENVIRONMENT_NAME}
```

### 1.1 Using pip
Please use the following command to install TorchSketch. 
```bash
pip install torchsketch
```
Then, TorchSketch can be imported into your Python console as follows.
```python
import torchsketch
```

### 1.2 From Source
In addition, TorchSketch also can be installed from source.
```bash
# Choose your workspace and download this repository.
cd ${CUSTOMIZED_WORKSPACE}
git clone https://github.com/PengBoXiangShang/torchsketch

# Enter the folder of TorchSketch.
cd torchsketch

# Install.
python setup.py install
```


## 2. Major Modules and Features of TorchSketch

### 2.1 Major Modules
TorchSketch has three main modules, including `data`, `networks`, `utils`, as shown in follows.
The documents are provided in `docs`.
  - **torchsketch**
    - **data**
      - **dataloaders**: provides the dataloader class files for the frequently-used sketch datasets, e.g., TU-Berlin, Sketchy, QuickDraw.
      - **datasets**: provides the specific API for each dataset, which integrates a series of functions including downloading, extraction, cleaning, MD5 checksum, and other preprocessings.
    - **networks**
      - **cnn**: provides all the SOTA CNNs.
      - **gnn**: provides the sketch-applicable implementations of GNNs, including GCN, GAT, graph transformer, etc.
      - **rnn**: provides the sketch-applicable implementations of RNNs.
      - **tcn**: provides the sketch-applicable implementations of TCNs.
    - **utils**
      - **data_augmentation_utils**
      - **general_utils**
      - **metric_utils**
      - **self_supervised_utils**
      - **svg_specific_utils**
    - **docs**
      - **api_reference**
      - **examples**

These modules and sub-modules can be imported as follows.
```python
import torchsketch.data.dataloaders as dataloaders
import torchsketch.data.datasets as datasets

import torchsketch.networks.cnn as cnns
import torchsketch.networks.gnn as gnns
import torchsketch.networks.rnn as rnns
import torchsketch.networks.tcn as tcns

import torchsketch.utils.data_augmentation_utils as data_augmentation_utils
import torchsketch.utils.general_utils as general_utils
import torchsketch.utils.metric_utils as metric_utils
import torchsketch.utils.self_supervised_utils as self_supervised_utils
import torchsketch.utils.svg_specific_utils as svg_specific_utils
```


### 2.2 Major Features
  - TorchSketch supports both GPU based and Python built-in multi-processing acceleration.
  - TorchSketch is modular, flexible, and extensible, without overly complex design patterns and excessive encapsulation.
  - TorchSketch provides four kinds of network architectures that are applicable to sketch, i.e., CNN, RNN, GNN, TCN.
  - TorchSketch is compatible to not only numerous datasets but also various formats of free-hand sketch, e.g., SVG, NumPy, PNG, JPEG, by providing numerous format-convert APIs, format-specific APIs, etc.
  - TorchSketch supports self-supervised learning study for sketch.
  - TorchSketch, beyond free-hand sketch research, also has some universal components that are applicable to the studies for other deep learning topics.

