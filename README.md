# ACCycleGAN
This is the code (in PyTorch) for our paper “[Semantic-aware Automatic Image Colorization via Unpaired
Cycle-Consistent Self-supervised Network](http://doi.org/10.1002/int.22667)”，accepted in *International Jounral of Intelligent Systems*.

## Prerequisites
Linux

Python 3

CPU or NVIDIA GPU + CUDA CuDNN

## Datasets

The color domain data  in the paper is randomly selected from the [PASCAL VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/), and grayscaled color domain data to gray domain data.
You can build your own dataset by setting up the following directory structure:

    ├── data                 
    |   ├── src_data         # gray
    |   |   ├── JPEGImages
    |   |   ├── SegmentationClass 
    |   ├── tgt_data         # color
    |   |   ├── JPEGImages 
    |   |   ├── SegmentationClass

## Running 
- For train
```
python colorization.py
```
- For test
```
python test.py
```

## Reference
If you find the code useful, please cite:
```
@article{https://doi.org/10.1002/int.22667,
author = {Xiao, Yuxuan and Jiang, Aiwen and Liu, Changhong and Wang, Mingwen},
title = {Semantic-aware automatic image colorization via unpaired cycle-consistent self-supervised network},
journal = {International Journal of Intelligent Systems},
volume = {37},
number = {2},
pages = {1222-1238},
keywords = {CycleGAN, image colorization, image editing, unpaired training, unsupervised learning},
doi = {https://doi.org/10.1002/int.22667},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/int.22667},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/int.22667},
year = {2022}
}
  ```
