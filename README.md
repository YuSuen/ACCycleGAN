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
