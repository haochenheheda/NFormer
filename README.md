# NFormer

Implementation of NFormer: Robust Person Re-identification with Neighbor Transformer. CVPR2022
Picture

## Requirements
 - Python3
 - pytorch>=0.4
 - torchvision
 - ignite=0.1.2 (Note: V0.2.0 may result in an error)
 - yacs

## Dataset
Create a directory to store reid datasets under this repo or outside this repo. Set your path to the root of the dataset in `config/defaults.py` or set in scripts `Experiment-all_tricks-tri_center-market.sh` and `x`.

### Market1501



## Training

## Evaluation
 

## Acknowledgement
This repo is highly based on [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline), thanks for their excellent work.

## Citation
```
@article{wang2022nformer,
  title={NFormer: Robust Person Re-identification with Neighbor Transformer},
  author={Wang, Haochen and Shen, Jiayi and Liu, Yongtuo and Gao, Yan and Gavves, Efstratios},
  journal={arXiv preprint arXiv:2204.09331},
  year={2022}
}

@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```



