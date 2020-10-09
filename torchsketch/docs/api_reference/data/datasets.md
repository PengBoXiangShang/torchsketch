# torchsketch.data.datasets


## 1. Purposes
**torchsketch.data.datasets** submodule provides the functions for the frequently-used free-hand sketch datasets (e.g., TU-Berlin, Sketchy, QuickDraw), integrating a series of functions including downloading, extraction, cleaning, MD5 checksum, and other preprocessing.


## 2. Function/API Organization
  - **torchsketch.data.datasets**
    - download_qmul_chair(output_folder = "./", remove_sourcefile = True), [[source]](https://github.com/PengBoXiangShang/torchsketch/blob/master/torchsketch/data/datasets/qmul_chair/download_qmul_chair.py)
    - download_qmul_shoe(output_folder = "./", remove_sourcefile = True), [[source]](https://github.com/PengBoXiangShang/torchsketch/blob/master/torchsketch/data/datasets/qmul_shoe/download_qmul_shoe.py)
    - download_quickdraw_414k(output_folder = "./", remove_sourcefile = True), [[source]](https://github.com/PengBoXiangShang/torchsketch/blob/master/torchsketch/data/datasets/quickdraw/quickdraw_414k/download_quickdraw_414k.py)
    - download_sketchy(output_folder = "./", remove_sourcefile = True), [[source]](https://github.com/PengBoXiangShang/torchsketch/blob/master/torchsketch/data/datasets/sketchy/download_sketchy.py)
    - download_tu_berlin(output_folder = "./", remove_sourcefile = True), [[source]](https://github.com/PengBoXiangShang/torchsketch/blob/master/torchsketch/data/datasets/tu_berlin/download_tu_berlin.py)


## 3. Examples 
```
import torchsketch.data.datasets as datasets

datasets.download_qmul_chair()

datasets.download_qmul_shoe()

datasets.download_quickdraw_414k()

datasets.download_sketchy()

datasets.download_tu_berlin()
datasets.download_tu_berlin(output_folder = YOUR_TARGET_PATH)
datasets.download_tu_berlin(output_folder = YOUR_TARGET_PATH, remove_sourcefile = False)
```
