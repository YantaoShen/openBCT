# OpenBCT

## Introduction

`OpenBCT` is a Non Official Reimplementation for [Towards Backward-Compatible Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html) (CVPR 2020 oral), which aims at achieving compatibility across different models.

### Feature List

- [x] Baselines for Backward Compatible Training
- [x] Influence Loss with old classifier
- [x] Pseudo old classifier generation

## Requirements

* Python >= 3.6
* Pytorch >= 1.2.0
* torchvision == 0.2.1
* numpy
* sklearn
* easydict

## Datasets

This code conducts training and evaluation on [ImageNet LSVRC 2012](http://image-net.org/challenges/LSVRC/2012/) and [Places365](http://places2.csail.mit.edu/download.html)(easy directory structure) datasets. Please be noticed that, due to the privacy issue, this code will NOT provide the train and test code for face datasets in our CVPR 2020 paper.

After you download these two datasets, extract them and make sure there are ./train and ./val folders inside. You can find the training image lists we use from [Google Drive Link](https://drive.google.com/drive/folders/169L3zAxJj_e1_BAZNvvR4IIy0cdbtC4E?usp=sharing) or [Weiyun Link](https://share.weiyun.com/zbdtIhlO).

## Train

For training an old model without any regularization,

```shell
python main.py your_dataset_dir --train-img-list imgnet_train_img_list_for_old.txt -a resnet18 
```

For training a new model with infulence loss (old classifier regularization),

```shell
python main.py your_dataset_dir --train-img-list imgnet_train_img_list_for_new.txt -a resnet50 --old-fc your_old_fc_weights_dir --n2o-map ./imgnet_new2old_map.npy
```

For training a new model with L2 regression loss (one of the compared baseline),

```shell
python main.py your_dataset_dir --train-img-list imgnet_train_img_list_for_new.txt -a resnet50 --old-arch resnet18 --old-checkpoint your_old_model_dir --old-fc your_old_fc_weights_dir --l2 --use-feat
```

## Test

For cross test between two models,

```shell
python main.py your_dataset_dir -a resnet50 --pretrained --checkpoint your_new_model_dir --old-fc your_old_fc_weights_dir --use-feat -e --cross-eval --old-arch resnet18 --old-checkpoint your_old_model_dir 
```

For self test with single model,

```shell
python main.py your_dataset_dir -a resnet50 --pretrained --checkpoint your_model_dir --use-feat -e
```

## Results

Influence loss results on ImagNet are listed as below. Please be noted that, in testing, we use feature L2 distance for Top-k accuracy computing. More results are on the way.

| Old Model | New Model | Val Set | Top-1 Acc | Top-5 Acc |
| :-------: | :-------: | :-----: | :-------: | :-------: |
|resnet18 (50%) | resnet18 (50%) | ImageNet Val |     39.5%     |  60.0%      |
|    resnet18 (50%)     |    resnet50-BCT (100%)      |    ImageNet Val    | 42.2% | 65.5% |
| resnet50 (100%) | resnet50 (100%) | ImageNet Val | 62.5% | 81.5% |
| resnet18 (50%) | resnet18 (50%) | Places365 Val | 27.0% | 55.9% |
| resnet18 (50%) | resnet50-BCT (100%) | Places365 Val | 27.5% | 57.8% |
| resnet50 (100%) | resnet50 (100%) | Places365 Val | 35.1% | 64.0% |

In this table, x% denotes the training data usage amount. 

## Acknowledgement

The code is based on [Open-ReID](https://github.com/Cysu/open-reid) and [Pytorch-ImageNet-Example](https://github.com/pytorch/examples/tree/master/imagenet). Thank these researchers for sharing their great code! 

## Citation

If this code helps your research or project, please cite

```
@inproceedings{shen2020towards,
  title={Towards backward-compatible representation learning},
  author={Shen, Yantao and Xiong, Yuanjun and Xia, Wei and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6368--6377},
  year={2020}
}
```

## Contact

If you have any question, please feel free to contact

```
Yantao Shen: ytshen@link.cuhk.edu.hk
```





