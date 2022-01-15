# [Robust 3D Hand Detection from a Single RGB-D Image in Unconstrained Environments](https://www.mdpi.com/1424-8220/20/21/6360)
## Introduction
This is the official implementation for the paper, "**Robust 3D Hand Detection from a Single RGB-D Image in Unconstrained Environments**", Sensors 2020.

## Environment
The code is developed using python 3.8 on Ubuntu 20.04. The code is developed and tested using NVIDIA 2080Ti GPU card.

## Quick start
### installation
1. install pytorch following [official instruction](https://pytorch.org/). Tested with pytorch v1.7.0
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi)

### Data preparation
We release two version of **CUG Hand data**, raw version and coco version.
The only difference between these two version is the data arangement and annotation format, They share the same number of images and bounding box annotations.
Our code is based on [raw version](https://1drv.ms/u/s!AiO4RSFtV9keh0xt9_jUD1_2GwLc?e=K9jesV).
For convenience, [coco version](https://1drv.ms/u/s!AiO4RSFtV9kehm4ULtjAWdeCWJXn?e=ctbeAP) is compatible with coco api.

### Training
```
python hand_detection/train.py
```

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{Xu_2020, 
  title={Robust 3D Hand Detection from a Single RGB-D Image in Unconstrained Environments}, 
  author={Xu, Chi and Zhou, Jun and Cai, Wendi and Jiang, Yunkai and Li, Yongbo and Liu, Yi}, 
  year={2020}, 
  month={Nov}, 
  pages={6360},
  volume={20}, 
  ISSN={1424-8220}, 
  url={http://dx.doi.org/10.3390/s20216360}, 
  DOI={10.3390/s20216360}, 
  number={21}, 
  journal={Sensors}, 
  publisher={MDPI AG}
}

```
