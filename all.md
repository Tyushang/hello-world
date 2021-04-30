# Probabilistic Anchor Assignment with IoU Prediction for Object Detection

<!-- [ALGORITHM] -->

```latex
@inproceedings{paa-eccv2020,
  title={Probabilistic Anchor Assignment with IoU Prediction for Object Detection},
  author={Kim, Kang and Lee, Hee Seok},
  booktitle = {ECCV},
  year={2020}
}
```

## Results and Models

We provide config files to reproduce the object detection results in the
ECCV 2020 paper for Probabilistic Anchor Assignment with IoU
Prediction for Object Detection.

| Backbone    | Lr schd | Mem (GB) | Score voting | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:------------:|:------:|:------:|:--------:|
| R-50-FPN    | 12e     | 3.7     | True          | 40.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1x_coco/paa_r50_fpn_1x_coco_20200821-936edec3.log.json) |
| R-50-FPN    | 12e     | 3.7     | False         | 40.2   | - |
| R-50-FPN    | 18e     | 3.7     | True          | 41.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_1.5x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_1.5x_coco/paa_r50_fpn_1.5x_coco_20200823-805d6078.log.json) |
| R-50-FPN    | 18e     | 3.7     | False         | 41.2   | - |
| R-50-FPN    | 24e     | 3.7     | True          | 41.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_2x_coco/paa_r50_fpn_2x_coco_20200821-c98bfc4e.log.json) |
| R-50-FPN    | 36e     | 3.7     | True          | 43.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r50_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722-06a6880b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r50_fpn_mstrain_3x_coco/paa_r50_fpn_mstrain_3x_coco_20210121_145722.log.json) |
| R-101-FPN   | 12e     | 6.2     | True          | 42.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.log.json) |
| R-101-FPN   | 12e     | 6.2     | False         | 42.4   | - |
| R-101-FPN   | 24e     | 6.2     | True          | 43.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_2x_coco/paa_r101_fpn_2x_coco_20200821-6829f96b.log.json) |
| R-101-FPN   | 36e     | 6.2     | True          | 45.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/paa/paa_r101_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202-83250d22.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_mstrain_3x_coco/paa_r101_fpn_mstrain_3x_coco_20210122_084202.log.json) |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.2 mAP. We report the best results.
# Gradient Harmonized Single-stage Detector

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{li2019gradient,
  title={Gradient Harmonized Single-stage Detector},
  author={Li, Buyu and Liu, Yu and Wang, Xiaogang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
```

## Results and Models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    |   4.0    | 3.3            |  37.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130-a437fda3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r50_fpn_1x_coco/retinanet_ghm_r50_fpn_1x_coco_20200130_004213.log.json) |
|    R-101-FPN    | pytorch |   1x    |   6.0    | 4.4            |  39.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130-c148ee8f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_r101_fpn_1x_coco/retinanet_ghm_r101_fpn_1x_coco_20200130_145259.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.2    | 5.1            |  40.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131-e4333bd0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_32x4d_fpn_1x_coco/retinanet_ghm_x101_32x4d_fpn_1x_coco_20200131_113653.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.3   | 5.2            |  41.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131-dd381cef.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ghm/retinanet_ghm_x101_64x4d_fpn_1x_coco/retinanet_ghm_x101_64x4d_fpn_1x_coco_20200131_113723.log.json) |
# Group Normalization

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{wu2018group,
  title={Group Normalization},
  author={Wu, Yuxin and He, Kaiming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

## Results and Models

| Backbone      | model      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN (d)  | Mask R-CNN | 2x      | 7.1      | 11.0           | 40.2   | 36.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206-8eee02a6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_2x_coco/mask_rcnn_r50_fpn_gn-all_2x_coco_20200206_050355.log.json) |
| R-50-FPN (d)  | Mask R-CNN | 3x      | 7.1      | -              | 40.5   | 36.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214-8b23b1e5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_3x_coco/mask_rcnn_r50_fpn_gn-all_3x_coco_20200214_063512.log.json) |
| R-101-FPN (d) | Mask R-CNN | 2x      | 9.9      | 9.0            | 41.9   | 37.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r101_fpn_gn-all_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205-d96b1b50.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_2x_coco/mask_rcnn_r101_fpn_gn-all_2x_coco_20200205_234402.log.json) |
| R-101-FPN (d) | Mask R-CNN | 3x      | 9.9      |                | 42.1   | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r101_fpn_gn-all_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609-0df864f4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r101_fpn_gn-all_3x_coco/mask_rcnn_r101_fpn_gn-all_3x_coco_20200513_181609.log.json) |
| R-50-FPN (c)  | Mask R-CNN | 2x      | 7.1      | 10.9           | 40.0   | 36.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207-20d3e849.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco/mask_rcnn_r50_fpn_gn-all_contrib_2x_coco_20200207_225832.log.json) |
| R-50-FPN (c)  | Mask R-CNN | 3x      | 7.1      | -              | 40.1   | 36.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225-542aefbc.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco/mask_rcnn_r50_fpn_gn-all_contrib_3x_coco_20200225_235135.log.json) |

**Notes:**

- (d) means pretrained model converted from Detectron, and (c) means the contributed model pretrained by [@thangvubk](https://github.com/thangvubk).
- The `3x` schedule is epoch [28, 34, 36].
- **Memory, Train/Inf time is outdated.**
# RepPoints: Point Set Representation for Object Detection

By [Ze Yang](https://yangze.tech/), [Shaohui Liu](http://b1ueber2y.me/), and [Han Hu](https://ancientmooner.github.io/).

We provide code support and configuration files to reproduce the results in the paper for
["RepPoints: Point Set Representation for Object Detection"](https://arxiv.org/abs/1904.11490) on COCO object detection.

## Introduction

<!-- [ALGORITHM] -->

**RepPoints**, initially described in [arXiv](https://arxiv.org/abs/1904.11490), is a new representation method for visual objects, on which visual understanding tasks are typically centered. Visual object representation, aiming at both geometric description and appearance feature extraction, is conventionally achieved by `bounding box + RoIPool (RoIAlign)`. The bounding box representation is convenient to use; however, it provides only a rectangular localization of objects that lacks geometric precision and may consequently degrade feature quality. Our new representation, RepPoints, models objects by a `point set` instead of a `bounding box`, which learns to adaptively position themselves over an object in a manner that circumscribes the object’s `spatial extent` and enables `semantically aligned feature extraction`. This richer and more flexible representation maintains the convenience of bounding boxes while facilitating various visual understanding applications. This repo demonstrated the effectiveness of RepPoints for COCO object detection.

Another feature of this repo is the demonstration of an `anchor-free detector`, which can be as effective as state-of-the-art anchor-based detection methods. The anchor-free detector can utilize either `bounding box` or `RepPoints` as the basic object representation.

<div align="center">
  <img src="reppoints.png" width="400px" />
  <p>Learning RepPoints in Object Detection.</p>
</div>

## Citing RepPoints

```
@inproceedings{yang2019reppoints,
  title={RepPoints: Point Set Representation for Object Detection},
  author={Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the table below.

| Method    | Backbone      | GN  | Anchor | convert func | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------------:|:---:|:------:|:------------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| BBox      | R-50-FPN      | Y   | single | -            | 1x      | 3.9      | 15.9           | 36.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/bbox_r50_grid_fpn_gn-neck+head_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329-c98bfa96.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_fpn_gn-neck%2Bhead_1x_coco_20200329_145916.log.json) |
| BBox      | R-50-FPN      | Y   | none   | -            | 1x      | 3.9      | 15.4           | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/bbox_r50_grid_center_fpn_gn-neck+Bhead_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco_20200330-00f73d58.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco/bbox_r50_grid_center_fpn_gn-neck%2Bhead_1x_coco_20200330_233609.log.json) |
| RepPoints | R-50-FPN      | N   | none   | moment       | 1x      | 3.3      | 18.5           | 37.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330-b73db8d1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_1x_coco/reppoints_moment_r50_fpn_1x_coco_20200330_233609.log.json) |
| RepPoints | R-50-FPN      | Y   | none   | moment       | 1x      | 3.9      | 17.5           | 38.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329-4b38409a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_1x_coco_20200329_145952.log.json) |
| RepPoints | R-50-FPN      | Y   | none   | moment       | 2x      | 3.9      | -              | 38.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329-91babaa2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r50_fpn_gn-neck%2Bhead_2x_coco_20200329_150020.log.json) |
| RepPoints | R-101-FPN     | Y   | none   | moment       | 2x      | 5.8      | 13.7           | 40.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r101_fpn_gn-neck+head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329-4fbc7310.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_gn-neck%2Bhead_2x_coco_20200329_132205.log.json) |
| RepPoints | R-101-FPN-DCN | Y   | none   | moment       | 2x      | 5.9      | 12.1           | 42.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-3309fbf2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329_132134.log.json) |
| RepPoints | X-101-FPN-DCN | Y   | none   | moment       | 2x      | 7.1      | 9.3            | 44.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329_132201.log.json) |

**Notes:**

- `R-xx`, `X-xx` denote the ResNet and ResNeXt architectures, respectively.
- `DCN` denotes replacing 3x3 conv with the 3x3 deformable convolution in `c3-c5` stages of backbone.
- `none` in the `anchor` column means 2-d `center point` (x,y) is used to represent the initial object hypothesis. `single` denotes one 4-d anchor box (x,y,w,h) with IoU based label assign criterion is adopted.
- `moment`, `partial MinMax`, `MinMax` in the `convert func` column are three functions to convert a point set to a pseudo box.
- Note the results here are slightly different from those reported in the paper, due to framework change. While the original paper uses an [MXNet](https://mxnet.apache.org/) implementation, we re-implement the method in [PyTorch](https://pytorch.org/) based on mmdetection.
# GRoIE

## A novel Region of Interest Extraction Layer for Instance Segmentation

By Leonardo Rossi, Akbar Karimi and Andrea Prati from
[IMPLab](http://implab.ce.unipr.it/).

We provide configs to reproduce the results in the paper for
"*A novel Region of Interest Extraction Layer for Instance Segmentation*"
on COCO object detection.

## Introduction

<!-- [ALGORITHM] -->

This paper is motivated by the need to overcome to the limitations of existing
RoI extractors which select only one (the best) layer from FPN.

Our intuition is that all the layers of FPN retain useful information.

Therefore, the proposed layer (called Generic RoI Extractor - **GRoIE**)
introduces non-local building blocks and attention mechanisms to boost the
performance.

## Results and models

The results on COCO 2017 minival (5k images) are shown in the below table.
You can find
[here](https://drive.google.com/drive/folders/19ssstbq_h0Z1cgxHmJYFO8s1arf3QJbT)
the trained models.

### Application of GRoIE to different architectures

| Backbone  | Method            | Lr schd | box AP | mask AP |  Config | Download|
| :-------: | :--------------: | :-----: | :----: | :-----: | :-------:| :--------:|
| R-50-FPN  | Faster Original  |   1x    |  37.4  |         | [config](../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |  38.3  |         | [config](./faster_rcnn_r50_fpn_groie_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715-66ee9516.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/faster_rcnn_r50_fpn_groie_1x_coco/faster_rcnn_r50_fpn_groie_1x_coco_20200604_211715.log.json) |
| R-50-FPN  | Grid R-CNN       |   1x    |  39.1  |         | [config](./grid_rcnn_r50_fpn_gn-head_1x_coco.py)| [model](http://download.openmmlab.com/mmdetection/v2.0/groie/grid_rcnn_r50_fpn_gn-head_1x_coco/grid_rcnn_r50_fpn_gn-head_1x_coco_20200605_202059-64f00ee8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/grid_rcnn_r50_fpn_gn-head_1x_coco/grid_rcnn_r50_fpn_gn-head_1x_coco_20200605_202059.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |    |         | [config](./grid_rcnn_r50_fpn_gn-head_groie_1x_coco.py)||
| R-50-FPN  | Mask R-CNN       |   1x    |  38.2  |  34.7   | [config](../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)| [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |  39.0  |  36.0   | [config](./mask_rcnn_r50_fpn_groie_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_groie_1x_coco/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715-50d90c74.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_groie_1x_coco/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715.log.json) |
| R-50-FPN  | GC-Net           |   1x    |  40.7  |  36.5   | [config](../gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202_085547.log.json) |
| R-50-FPN  | + GRoIE          |   1x    |  41.0  |  37.8   | [config](./mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200604_211715-42eb79e1.pth) |
| R-101-FPN | GC-Net           |   1x    |  42.2  |  37.8   | [config](../gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206_142508.log.json) |
| R-101-FPN | + GRoIE          |   1x    |   |    | [config](./mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py)| [model](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507-8daae01c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/groie/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco_20200607_224507.log.json) |

## Citation

If you use this work or benchmark in your research, please cite this project.

```latex
@misc{rossi2020novel,
    title={A novel Region of Interest Extraction Layer for Instance Segmentation},
    author={Leonardo Rossi and Akbar Karimi and Andrea Prati},
    year={2020},
    eprint={2004.13665},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Contact

The implementation of GROI is currently maintained by
[Leonardo Rossi](https://github.com/hachreak/).
# DetectoRS

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf).

```BibTeX
@article{qiao2020detectors,
  title={DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution},
  author={Qiao, Siyuan and Chen, Liang-Chieh and Yuille, Alan},
  journal={arXiv preprint arXiv:2006.02334},
  year={2020}
}
```

## Dataset

DetectoRS requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

DetectoRS includes two major components:

- Recursive Feature Pyramid (RFP).
- Switchable Atrous Convolution (SAC).

They can be used independently.
Combining them together results in DetectoRS.
The results on COCO 2017 val are shown in the below table.

| Method | Detector | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------:|:--------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| RFP | Cascade + ResNet-50 | 1x | 7.5 | - | 44.8 | | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/cascade_rcnn_r50_rfp_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_rfp_1x_coco/cascade_rcnn_r50_rfp_1x_coco_20200624_104126.log.json) |
| SAC | Cascade + ResNet-50 | 1x | 5.6 | - | 45.0| | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/cascade_rcnn_r50_sac_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco-24bfda62.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/cascade_rcnn_r50_sac_1x_coco/cascade_rcnn_r50_sac_1x_coco_20200624_104402.log.json) |
| DetectoRS | Cascade + ResNet-50 | 1x | 9.9 | - | 47.4 | | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco_20200706_001203.log.json) |
| RFP | HTC + ResNet-50 | 1x | 11.2 | - | 46.6 | 40.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/htc_r50_rfp_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco-8ff87c51.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_rfp_1x_coco/htc_r50_rfp_1x_coco_20200624_103053.log.json) |
| SAC | HTC + ResNet-50 | 1x | 9.3 | - | 46.4 | 40.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/htc_r50_sac_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco-bfa60c54.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/htc_r50_sac_1x_coco/htc_r50_sac_1x_coco_20200624_103111.log.json) |
| DetectoRS | HTC + ResNet-50 | 1x | 13.6 | - | 49.1 | 42.6 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detectors/detectors_htc_r50_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco_20200624_103659.log.json) |

*Note*: This is a re-implementation based on MMDetection-V2.
The original implementation is based on MMDetection-V1.
# CentripetalNet

## Introduction

<!-- [ALGORITHM] -->

```latex
@InProceedings{Dong_2020_CVPR,
author = {Dong, Zhiwei and Li, Guoxuan and Liao, Yue and Wang, Fei and Ren, Pengju and Qian, Chen},
title = {CentripetalNet: Pursuing High-Quality Keypoint Pairs for Object Detection},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Results and models

| Backbone        | Batch Size | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :------: | :--------: |
| HourglassNet-104 | [16 x 6](./centripetalnet_hourglass104_mstest_16x6_210e_coco.py) | 190/210 | 16.7 | 3.7 | 44.8 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804.log.json) |

Note:

- TTA setting is single-scale and `flip=True`.
- The model we released is the best checkpoint rather than the latest checkpoint (box AP 44.8 vs 44.6 in our experiment).
# FreeAnchor: Learning to Match Anchors for Visual Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{zhang2019freeanchor,
  title   =  {{FreeAnchor}: Learning to Match Anchors for Visual Object Detection},
  author  =  {Zhang, Xiaosong and Wan, Fang and Liu, Chang and Ji, Rongrong and Ye, Qixiang},
  booktitle =  {Neural Information Processing Systems},
  year    =  {2019}
}
```

## Results and Models

| Backbone | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:--------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50     | pytorch | 1x      | 4.9      | 18.4 | 38.7 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130_095625.log.json) |
| R-101       | pytorch | 1x   | 6.8      | 14.9 | 40.3 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130-358324e6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r101_fpn_1x_coco/retinanet_free_anchor_r101_fpn_1x_coco_20200130_100723.log.json) |
| X-101-32x4d | pytorch | 1x   | 8.1      | 11.1 | 41.9 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130-d4846968.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_x101_32x4d_fpn_1x_coco/retinanet_free_anchor_x101_32x4d_fpn_1x_coco_20200130_095627.log.json) |

**Notes:**

- We use 8 GPUs with 2 images/GPU.
- For more settings and models, please refer to the [official repo](https://github.com/zhangxiaosong18/FreeAnchor).
# LVIS dataset

## Introduction

<!-- [DATASET] -->

```latex
@inproceedings{gupta2019lvis,
  title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
  author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Common Setting

* Please follow [install guide](../../docs/install.md#install-mmdetection) to install open-mmlab forked cocoapi first.
* Run following scripts to install our forked lvis-api.

    ```shell
    pip install git+https://github.com/lvis-dataset/lvis-api.git
    ```

* All experiments use oversample strategy [here](../../docs/tutorials/new_dataset.md#class-balanced-dataset) with oversample threshold `1e-3`.
* The size of LVIS v0.5 is half of COCO, so schedule `2x` in LVIS is roughly the same iterations as `1x` in COCO.

## Results and models of LVIS v0.5

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |:--------: |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 26.1   | 25.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis-dbd06831.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_20200531_160435.log.json)  |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 27.1   | 27.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis-54582ee2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_2x_lvis_20200601_134748.log.json)  |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 26.7   | 26.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis-3cf55ea2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_2x_lvis_20200531_221749.log.json)  |
| X-101-64x4d-FPN | pytorch |   2x    | -        |   -            | 26.4   | 26.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_v0.5.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis-1c99a5ad.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_2x_lvis_20200601_194651.log.json)  |

## Results and models of LVIS v1

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    | 9.1      | -              | 22.5   | 21.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_061305.log.json)  |
|    R-101-FPN    | pytorch |   1x    | 10.8     | -              | 24.6   | 23.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-ec55ce32.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_070959.log.json)  |
| X-101-32x4d-FPN | pytorch |   1x    | 11.8     | -              | 26.7   | 25.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-ebbc5c81.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-20200829_071317.log.json)  |
| X-101-64x4d-FPN | pytorch |   1x    | 14.6     | -              | 27.2   | 25.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-43d9edfe.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/lvis/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1/mask_rcnn_x101_64x4d_fpn_sample1e-3_mstrain_1x_lvis_v1-20200830_060206.log.json)  |
# Weight Standardization

## Introduction

<!-- [ALGORITHM] -->

```
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1903.10520},
  year      = {2019},
}
```

## Results and Models

Faster R-CNN

| Backbone  | Style   | Normalization | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN  | pytorch | GN+WS         | 1x      | 5.9      | 11.7           | 39.7   | -       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130_210936.log.json) |
| R-101-FPN | pytorch | GN+WS         | 1x      | 8.9      | 9.0            | 41.7   | -       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/faster_rcnn_r101_fpn_gn_ws-all_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r101_fpn_gn_ws-all_1x_coco/faster_rcnn_r101_fpn_gn_ws-all_1x_coco_20200205-a93b0d75.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r101_fpn_gn_ws-all_1x_coco/faster_rcnn_r101_fpn_gn_ws-all_1x_coco_20200205_232146.log.json) |
| X-50-32x4d-FPN | pytorch | GN+WS    | 1x      | 7.0      | 10.3           | 40.7   | -       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco_20200203-839c5d9d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x50_32x4d_fpn_gn_ws-all_1x_coco_20200203_220113.log.json) |
| X-101-32x4d-FPN | pytorch | GN+WS   | 1x      | 10.8     | 7.6            | 42.1   | -       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco_20200212-27da1bc2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco/faster_rcnn_x101_32x4d_fpn_gn_ws-all_1x_coco_20200212_195302.log.json) |

Mask R-CNN

| Backbone  | Style   | Normalization | Lr schd   | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------------:|:---------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN  | pytorch | GN+WS         | 2x        | 7.3      | 10.5       | 40.6        | 36.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco/mask_rcnn_r50_fpn_gn_ws-all_2x_coco_20200226-16acb762.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_2x_coco/mask_rcnn_r50_fpn_gn_ws-all_2x_coco_20200226_062128.log.json) |
| R-101-FPN | pytorch | GN+WS         | 2x        | 10.3     | 8.6        | 42.0        | 37.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_2x_coco/mask_rcnn_r101_fpn_gn_ws-all_2x_coco_20200212-ea357cd9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_2x_coco/mask_rcnn_r101_fpn_gn_ws-all_2x_coco_20200212_213627.log.json) |
| X-50-32x4d-FPN | pytorch | GN+WS    | 2x        | 8.4      | 9.3       | 41.1        | 37.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216-649fdb6f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_2x_coco_20200216_201500.log.json) |
| X-101-32x4d-FPN | pytorch | GN+WS   | 2x        | 12.2     | 7.1       | 42.1        | 37.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco_20200319-33fb95b5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco/mask_rcnn_x101_32x4d_fpn_gn_ws-all_2x_coco_20200319_104101.log.json) |
| R-50-FPN  | pytorch | GN+WS         | 20-23-24e | 7.3      | -        | 41.1        | 37.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco_20200213-487d1283.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r50_fpn_gn_ws-all_20_23_24e_coco_20200213_035123.log.json) |
| R-101-FPN | pytorch | GN+WS         | 20-23-24e | 10.3     | -        | 43.1        | 38.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213-57b5a50f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_r101_fpn_gn_ws-all_20_23_24e_coco_20200213_130142.log.json) |
| X-50-32x4d-FPN | pytorch | GN+WS    | 20-23-24e | 8.4      | -        | 42.1        | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200226-969bcb2c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_x50_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200226_093732.log.json) |
| X-101-32x4d-FPN | pytorch | GN+WS   | 20-23-24e | 12.2     | -        | 42.7        | 38.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200316-e6cd35ef.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco/mask_rcnn_x101_32x4d_fpn_gn_ws-all_20_23_24e_coco_20200316_013741.log.json) |

Note:

- GN+WS requires about 5% more memory than GN, and it is only 5% slower than GN.
- In the paper, a 20-23-24e lr schedule is used instead of 2x.
- The X-50-GN and X-101-GN pretrained models are also shared by the authors.
# Cascade RPN

<!-- [ALGORITHM] -->

We provide the code for reproducing experiment results of [Cascade RPN](https://arxiv.org/abs/1909.06720).

```
@inproceedings{vu2019cascade,
  title={Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution},
  author={Vu, Thang and Jang, Hyunjun and Pham, Trung X and Yoo, Chang D},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```

## Benchmark

### Region proposal performance

| Method | Backbone | Style | Mem (GB) | Train time (s/iter) | Inf time (fps) | AR 1000 |                Download                |
|:------:|:--------:|:-----:|:--------:|:-------------------:|:--------------:|:-------:|:--------------------------------------:|
|  CRPN  | R-50-FPN | caffe |     -    |          -          |        -       |   72.0  | [model](https://drive.google.com/file/d/1qxVdOnCgK-ee7_z0x6mvAir_glMu2Ihi/view?usp=sharing) |

### Detection performance

|     Method    |   Proposal  | Backbone |  Style  | Schedule | Mem (GB) | Train time (s/iter) | Inf time (fps) | box AP |                   Download                   |
|:-------------:|:-----------:|:--------:|:-------:|:--------:|:--------:|:-------------------:|:--------------:|:------:|:--------------------------------------------:|
|   Fast R-CNN  | Cascade RPN | R-50-FPN |  caffe  |    1x    |    -     |          -          |        -       |  39.9  | [model](https://drive.google.com/file/d/1NmbnuY5VHi8I9FE8xnp5uNvh2i-t-6_L/view?usp=sharing) |
|  Faster R-CNN | Cascade RPN | R-50-FPN |  caffe  |    1x    |    -     |          -          |        -       |  40.4  | [model](https://drive.google.com/file/d/1dS3Q66qXMJpcuuQgDNkLp669E5w1UMuZ/view?usp=sharing) |
# GCNet for Object Detection

By [Yue Cao](http://yue-cao.me), [Jiarui Xu](http://jerryxu.net), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en), Fangyun Wei, [Han Hu](https://sites.google.com/site/hanhushomepage/).

We provide config files to reproduce the results in the paper for
["GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond"](https://arxiv.org/abs/1904.11492) on COCO object detection.

## Introduction

<!-- [ALGORITHM] -->

**GCNet** is initially described in [arxiv](https://arxiv.org/abs/1904.11492). Via absorbing advantages of Non-Local Networks (NLNet) and Squeeze-Excitation Networks (SENet),  GCNet provides a simple, fast and effective approach for global context modeling, which generally outperforms both NLNet and SENet on major benchmarks for various recognition tasks.

## Citing GCNet

```latex
@article{cao2019GCNet,
  title={GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  author={Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
  journal={arXiv preprint arXiv:1904.11492},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the below table.

| Backbone  | Model            | Context        | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download  |
| :-------: | :--------------: | :------------: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
| R-50-FPN  | Mask             | GC(c3-c5, r16) |   1x    | 5.0      |               | 39.7   | 35.9    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915-187da160.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco_20200515_211915.log.json)   |
| R-50-FPN  | Mask             | GC(c3-c5, r4)  |   1x    | 5.1      | 15.0          | 39.9   | 36.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204-17235656.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco_20200204_024626.log.json) |
| R-101-FPN | Mask             | GC(c3-c5, r16) |   1x    | 7.6      | 11.4           | 41.3   | 37.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco_20200205-e58ae947.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_r16_gcb_c3-c5_1x_coco_20200205_192835.log.json) |
| R-101-FPN | Mask             | GC(c3-c5, r4)  |   1x    | 7.8      | 11.6           | 42.2   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco_20200206-af22dc9d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_r4_gcb_c3-c5_1x_coco_20200206_112128.log.json) |

| Backbone  | Model            | Context        | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download  |
| :-------: | :--------------: | :------------: | :-----: | :------: | :------------: | :----: | :-----: | :------: |  :-------: |
| R-50-FPN  | Mask             |      -         |   1x    | 4.4      | 16.6           | 38.4   | 34.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202-bb3eb55c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_1x_coco_20200202_214122.log.json) |
| R-50-FPN  | Mask             | GC(c3-c5, r16) |   1x    | 5.0      | 15.5           | 40.4   | 36.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202-587b99aa.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200202_174907.log.json) |
| R-50-FPN  | Mask             | GC(c3-c5, r4)  |   1x    | 5.1      | 15.1           | 40.7   | 36.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202-50b90e5c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200202_085547.log.json) |
| R-101-FPN | Mask             |      -         |   1x    | 6.4      | 13.3           | 40.5   | 36.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco_20200210-81658c8a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_1x_coco_20200210_220422.log.json) |
| R-101-FPN | Mask             | GC(c3-c5, r16) |   1x    | 7.6      | 12.0           | 42.2   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200207-945e77ca.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200207_015330.log.json) |
| R-101-FPN | Mask             | GC(c3-c5, r4)  |   1x    | 7.8      | 11.8           | 42.2   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206-8407a3f0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_r101_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200206_142508.log.json) |
| X-101-FPN | Mask             |      -         |   1x    | 7.6      | 11.3            | 42.4   | 37.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200211-7584841c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200211_054326.log.json) |
| X-101-FPN | Mask             | GC(c3-c5, r16) |   1x    | 8.8      | 9.8            | 43.5   | 38.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-cbed3d2c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211_164715.log.json) |
| X-101-FPN | Mask             | GC(c3-c5, r4)  |   1x    | 9.0      | 9.7            | 43.9   | 39.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200212-68164964.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200212_070942.log.json) |
| X-101-FPN | Cascade Mask     |      -         |   1x    | 9.2      | 8.4            | 44.7   | 38.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200310-d5ad2a5e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco_20200310_115217.log.json) |
| X-101-FPN | Cascade Mask     | GC(c3-c5, r16) |   1x    | 10.3     | 7.7            | 46.2   | 39.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211-10bf2463.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco_20200211_184154.log.json) |
| X-101-FPN | Cascade Mask     | GC(c3-c5, r4)  |   1x    | 10.6     |                | 46.4   |   40.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200703_180653-ed035291.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco_20200703_180653.log.json) |
| X-101-FPN | DCN Cascade Mask |      -         |   1x    |          |                | 44.9   |   38.9  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco_20200516_182249-680fc3f2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_1x_coco_20200516_182249.log.json)|
| X-101-FPN | DCN Cascade Mask | GC(c3-c5, r16) |   1x    |          |                | 44.6   |         |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco_20200516_015634-08f56b56.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r16_gcb_c3-c5_1x_coco_20200516_015634.log.json) |
| X-101-FPN | DCN Cascade Mask | GC(c3-c5, r4)  |   1x    |          |                | 45.7   |  39.5   |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20200518_041145-24cabcfd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_dconv_c3-c5_r4_gcb_c3-c5_1x_coco_20200518_041145.log.json)  |

**Notes:**

- The `SyncBN` is added in the backbone for all models in **Table 2**.
- `GC` denotes Global Context (GC) block is inserted after 1x1 conv of backbone.
- `DCN` denotes replace 3x3 conv with 3x3 Deformable Convolution in `c3-c5` stages of backbone.
- `r4` and `r16` denote ratio 4 and ratio 16 in GC block respectively.
# Mask Scoring R-CNN

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{huang2019msrcnn,
    title={Mask Scoring R-CNN},
    author={Zhaojin Huang and Lichao Huang and Yongchao Gong and Chang Huang and Xinggang Wang},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019},
}
```

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN      | caffe      | 1x      | 4.5      |                |  38.2  | 36.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848.log.json) |
| R-50-FPN      | caffe      | 2x      | -        | -              | 38.8   | 36.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco/ms_rcnn_r50_caffe_fpn_2x_coco_20200506_004738.log.json) |
| R-101-FPN     | caffe      | 1x      | 6.5      |                | 40.4   | 37.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.404__segm_mAP-0.376_20200506_004755-b9b12a37.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x_coco/ms_rcnn_r101_caffe_fpn_1x_coco_20200506_004755.log.json) |
| R-101-FPN     | caffe      | 2x      | -        | -              | 41.1   | 38.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_bbox_mAP-0.411__segm_mAP-0.381_20200506_011134-5f3cc74f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r101_caffe_fpn_2x_coco/ms_rcnn_r101_caffe_fpn_2x_coco_20200506_011134.log.json) |
| R-X101-32x4d  | pytorch    | 2x      | 7.9      | 11.0           | 41.8   | 38.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206-81fd1740.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_32x4d_fpn_1x_coco/ms_rcnn_x101_32x4d_fpn_1x_coco_20200206_100113.log.json) |
| R-X101-64x4d  | pytorch    | 1x      | 11.0     | 8.0            | 43.0   | 39.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206_091744.log.json) |
| R-X101-64x4d  | pytorch    | 2x      | 11.0     | 8.0            | 42.6   | 39.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308-02a445e2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_2x_coco/ms_rcnn_x101_64x4d_fpn_2x_coco_20200308_012247.log.json) |
# Feature Pyramid Grids

## Introduction

```latex
@article{chen2020feature,
  title={Feature pyramid grids},
  author={Chen, Kai and Cao, Yuhang and Loy, Chen Change and Lin, Dahua and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2004.03580},
  year={2020}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN.
All backbones are Resnet-50 in pytorch style.

| Method       | Neck        | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------------:|:-----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:-------:|:--------:|
| Faster R-CNN | FPG         | 50e     | 20.0     | -              | 42.2   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/faster_rcnn_r50_fpg_crop640_50e_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/faster_rcnn_r50_fpg_crop640_50e_coco-76220505.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg_crop640_50e_coco/20210218_223520.log.json) |
| Faster R-CNN | FPG-chn128  | 50e     | 11.9     | -              | 41.2   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/faster_rcnn_r50_fpg-chn128_crop640_50e_coco-24257de9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco/20210218_221412.log.json) |
| Mask R-CNN   | FPG         | 50e     | 23.2     | -              | 42.7   | 37.8    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/mask_rcnn_r50_fpg_crop640_50e_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/mask_rcnn_r50_fpg_crop640_50e_coco-c5860453.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg_crop640_50e_coco/20210222_205447.log.json) |
| Mask R-CNN   | FPG-chn128  | 50e     | 15.3     | -              | 41.7   | 36.9    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/mask_rcnn_r50_fpg-chn128_crop640_50e_coco-5c6ea10d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fpg/mask_rcnn_r50_fpg-chn128_crop640_50e_coco/20210223_025039.log.json) |
| RetinaNet    | FPG         | 50e     | 20.8     | -              | 40.5   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/retinanet_r50_fpg_crop640_50e_coco-46fdd1c6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg_crop640_50e_coco/20210225_143957.log.json) |
| RetinaNet    | FPG-chn128  | 50e     | 19.9     | -              | 40.3   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/retinanet_r50_fpg-chn128_crop640_50e_coco-5cf33c76.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fpg/retinanet_r50_fpg-chn128_crop640_50e_coco/20210225_184328.log.json) |

**Note**: Chn128 means to decrease the number of channels of features and convs from 256 (default) to 128 in
Neck and BBox Head, which can greatly decrease memory consumption without sacrificing much precision.
# Prime Sample Attention in Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{cao2019prime,
  title={Prime sample attention in object detection},
  author={Cao, Yuhang and Chen, Kai and Loy, Chen Change and Lin, Dahua},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Results and models

| PISA | Network | Backbone            | Lr schd | box AP | mask AP | Config | Download |
|:----:|:-------:|:-------------------:|:-------:|:------:|:-------:|:------:|:--------:|
| ×    | Faster R-CNN | R-50-FPN       | 1x      | 36.4   |         | - |
| √    | Faster R-CNN | R-50-FPN       | 1x      | 38.4   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_r50_fpn_1x_coco/pisa_faster_rcnn_r50_fpn_1x_coco_20200506_185619.log.json)  |
| ×    | Faster R-CNN | X101-32x4d-FPN | 1x      | 40.1   |         | - |
| √    | Faster R-CNN | X101-32x4d-FPN | 1x      | 41.9   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco-e4accec4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco/pisa_faster_rcnn_x101_32x4d_fpn_1x_coco_20200505_181503.log.json) |
| ×    | Mask   R-CNN | R-50-FPN       | 1x      | 37.3   | 34.2    | - |
| √    | Mask   R-CNN | R-50-FPN       | 1x      | 39.1   | 35.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_mask_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco-dfcedba6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_mask_rcnn_r50_fpn_1x_coco/pisa_mask_rcnn_r50_fpn_1x_coco_20200508_150500.log.json) |
| ×    | Mask   R-CNN | X101-32x4d-FPN | 1x      | 41.1   | 37.1    | - |
| √    | Mask   R-CNN | X101-32x4d-FPN | 1x      |        |         |   |
| ×    | RetinaNet    | R-50-FPN       | 1x      | 35.6   |         | - |
| √    | RetinaNet    | R-50-FPN       | 1x      | 36.9   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_retinanet_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco-76409952.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_r50_fpn_1x_coco/pisa_retinanet_r50_fpn_1x_coco_20200504_014311.log.json) |
| ×    | RetinaNet    | X101-32x4d-FPN | 1x      | 39.0   |         | - |
| √    | RetinaNet    | X101-32x4d-FPN | 1x      | 40.7   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco-a0c13c73.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_retinanet_x101_32x4d_fpn_1x_coco/pisa_retinanet_x101_32x4d_fpn_1x_coco_20200505_001404.log.json) |
| ×    | SSD300       | VGG16          | 1x      | 25.6   |         | - |
| √    | SSD300       | VGG16          | 1x      | 27.6   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_ssd300_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco-710e3ac9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd300_coco/pisa_ssd300_coco_20200504_144325.log.json) |
| ×    | SSD300       | VGG16          | 1x      | 29.3   |         | - |
| √    | SSD300       | VGG16          | 1x      | 31.8   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pisa/pisa_ssd512_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco-247addee.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pisa/pisa_ssd512_coco/pisa_ssd512_coco_20200508_131030.log.json)  |

**Notes:**

- In the original paper, all models are trained and tested on mmdet v1.x, thus results may not be exactly the same with this release on v2.0.
- It is noted PISA only modifies the training pipeline so the inference time remains the same with the baseline.
# Rethinking ImageNet Pre-training

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{he2018rethinking,
  title={Rethinking imagenet pre-training},
  author={He, Kaiming and Girshick, Ross and Doll{\'a}r, Piotr},
  journal={arXiv preprint arXiv:1811.08883},
  year={2018}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | box AP | mask AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN | R-50-FPN  | pytorch | 6x      | 40.7   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_bbox_mAP-0.407_20200201_193013-90813d01.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_faster_rcnn_r50_fpn_gn_6x_20200201_193013.log.json) |
| Mask R-CNN   | R-50-FPN  | pytorch | 6x      | 41.2   | 37.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_bbox_mAP-0.412__segm_mAP-0.374_20200201_193051-1e190a40.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/scratch/mask_rcnn_r50_fpn_gn-all_scratch_6x_coco/scratch_mask_rcnn_r50_fpn_gn_6x_20200201_193051.log.json)  |

Note:

- The above models are trained with 16 GPUs.
# Mask R-CNN

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{He_2017,
   title={Mask R-CNN},
   journal={2017 IEEE International Conference on Computer Vision (ICCV)},
   publisher={IEEE},
   author={He, Kaiming and Gkioxari, Georgia and Dollar, Piotr and Girshick, Ross},
   year={2017},
   month={Oct}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    | 4.3      |                | 38.0   | 34.4    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_20200504_231812.log.json) |
|    R-50-FPN     | pytorch |   1x    | 4.4      | 16.1           | 38.2   | 34.7    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 39.2   | 35.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_20200505_003907.log.json)  |
|    R-101-FPN    |  caffe  |   1x    |          |                | 40.4   | 36.4    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758-805e06c1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco/mask_rcnn_r101_caffe_fpn_1x_coco_20200601_095758.log.json)|
|    R-101-FPN    | pytorch |   1x    | 6.4      | 13.5           | 40.0   | 36.1    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 40.8   | 36.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_2x_coco/mask_rcnn_r101_fpn_2x_coco_20200505_071027.log.json)  |
| X-101-32x4d-FPN | pytorch |   1x    | 7.6      | 11.3           | 41.9   | 37.5    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205_034906.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 42.2   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.422__segm_mAP-0.378_20200506_004702-faef898c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco/mask_rcnn_x101_32x4d_fpn_2x_coco_20200506_004702.log.json)  |
| X-101-64x4d-FPN | pytorch |   1x    | 10.7     | 8.0            | 42.8   | 38.4    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201_124310.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |  -       |   -            |  42.7  |  38.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208.log.json)|
| X-101-32x8d-FPN | pytorch |   1x    |  -       |   -            |  42.8  |  38.3   | |

## Pre-trained Models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    [R-50-FPN](./mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco.py)     |  caffe  |   2x    | 4.3      |                | 40.3   | 36.5    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco_bbox_mAP-0.403__segm_mAP-0.365_20200504_231822-a75c98ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_2x_coco_20200504_231822.log.json)
|    [R-50-FPN](./mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py)     |  caffe  |   3x    | 4.3      |                | 40.8   | 37.0    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_20200504_163245.log.json)
|    [X-101-32x8d-FPN](./mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py)     |  pytorch  |   1x    | -     |       | 43.6 | 39.0 |
|    [X-101-32x8d-FPN](./mask_rcnn_x101_32x8d_fpn_mstrain-poly_3x_coco.py)     |  pytorch  |   3x    | -     |       | 44.0 | 39.3 |
# Focal Loss for Dense Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |   3.5    |      18.6      |  36.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_caffe_fpn_1x_coco/retinanet_r50_caffe_fpn_1x_coco_20200531_012518.log.json) |
|    R-50-FPN     | pytorch |   1x    |   3.8    |      19.0      |  36.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  37.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131_114738.log.json) |
|    R-101-FPN    |  caffe  |   1x    |   5.5    |      14.7      |  38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531-b428fa0f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_caffe_fpn_1x_coco/retinanet_r101_caffe_fpn_1x_coco_20200531_012536.log.json) |
|    R-101-FPN    | pytorch |   1x    |   5.7    |      15.0      |  38.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_1x_coco/retinanet_r101_fpn_1x_coco_20200130_003055.log.json) |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131_114859.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    |      12.1      |  39.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_1x_coco/retinanet_x101_32x4d_fpn_1x_coco_20200130_003004.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  40.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_32x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131-237fc5e1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_32x4d_fpn_2x_coco/retinanet_x101_32x4d_fpn_2x_coco_20200131_114812.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.0   |      8.7       |  41.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130_003008.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  40.8  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131_114833.log.json) |
# Path Aggregation Network for Instance Segmentation

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{liu2018path,
  author = {Shu Liu and
            Lu Qi and
            Haifang Qin and
            Jianping Shi and
            Jiaya Jia},
  title = {Path Aggregation Network for Instance Segmentation},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```

## Results and Models

## Results and Models

| Backbone      | style      | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:-------------:|:----------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN      | pytorch    | 1x      | 4.0      | 17.2           | 37.5   |         | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pafpn/faster_rcnn_r50_pafpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_bbox_mAP-0.375_20200503_105836-b7b4b9bd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pafpn/faster_rcnn_r50_pafpn_1x_coco/faster_rcnn_r50_pafpn_1x_coco_20200503_105836.log.json) |
# Res2Net for object detection and instance segmentation

## Introduction

<!-- [ALGORITHM] -->

We propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer.

|    Backbone     |Params. | GFLOPs  | top-1 err. | top-5 err. |
| :-------------: |:----:  | :-----: | :--------: | :--------: |
| ResNet-101      |44.6 M  | 7.8     |  22.63     |  6.44      |
| ResNeXt-101-64x4d |83.5M | 15.5    |  20.40     |  -         |
| HRNetV2p-W48    | 77.5M  | 16.1    |  20.70     |  5.50      |
| Res2Net-101     | 45.2M  | 8.3     |  18.77     |  4.64      |

Compared with other backbone networks, Res2Net requires fewer parameters and FLOPs.

**Note:**

- GFLOPs for classification are calculated with image size (224x224).

```latex
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2020},
  doi={10.1109/TPAMI.2019.2938758},
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|R2-101-FPN       | pytorch |   2x   |   7.4    |   -           |  43.0  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/faster_rcnn_r2_101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco-175f1da6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/res2net/faster_rcnn_r2_101_fpn_2x_coco/faster_rcnn_r2_101_fpn_2x_coco_20200514_231734.log.json) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|R2-101-FPN       | pytorch |    2x   |   7.9    |      -         |   43.6 | 38.7  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/mask_rcnn_r2_101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/res2net/mask_rcnn_r2_101_fpn_2x_coco/mask_rcnn_r2_101_fpn_2x_coco-17f061e8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/res2net/mask_rcnn_r2_101_fpn_2x_coco/mask_rcnn_r2_101_fpn_2x_coco_20200515_002413.log.json) |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|R2-101-FPN       | pytorch |   20e   |   7.8    |      -         |  45.7  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco-f4b7b7db.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_rcnn_r2_101_fpn_20e_coco/cascade_rcnn_r2_101_fpn_20e_coco_20200515_091644.log.json) |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
R2-101-FPN       | pytorch |  20e   |    9.5  |      -         |  46.4  |  40.0  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco-8a7b41e1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco/cascade_mask_rcnn_r2_101_fpn_20e_coco_20200515_091645.log.json) |

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
| R2-101-FPN     | pytorch |   20e   |    -    |      -         |  47.5  | 41.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/res2net/htc_r2_101_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/res2net/htc_r2_101_fpn_20e_coco/htc_r2_101_fpn_20e_coco-3a8d2112.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/res2net/htc_r2_101_fpn_20e_coco/htc_r2_101_fpn_20e_coco_20200515_150029.log.json) |

- Res2Net ImageNet pretrained models are in [Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels).
- More applications of Res2Net are in [Res2Net-Github](https://github.com/Res2Net/).
# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle={Advances in neural information processing systems},
  year={2015}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR1000 | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |   3.5    |      22.6      |  58.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_caffe_fpn_1x_coco/rpn_r50_caffe_fpn_1x_coco_20200531-5b903a37.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_caffe_fpn_1x_coco/rpn_r50_caffe_fpn_1x_coco_20200531_012334.log.json) |
|    R-50-FPN     | pytorch |   1x    |   3.8    |      22.3      |  58.2  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218_151240.log.json) |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  58.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131_190631.log.json) |
|    R-101-FPN    |  caffe  |   1x    |   5.4    |      17.3      |  60.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_caffe_fpn_1x_coco/rpn_r101_caffe_fpn_1x_coco_20200531-0629a2e2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_caffe_fpn_1x_coco/rpn_r101_caffe_fpn_1x_coco_20200531_012345.log.json) |
|    R-101-FPN    | pytorch |   1x    |   5.8    |      16.5      |  59.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131_191000.log.json) |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  60.2  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131_191106.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.0    |      13.0      |  60.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219_012037.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  61.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_32x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208_200752.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |   10.1   |      9.1       |  61.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208_200752.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  61.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/rpn/rpn_x101_64x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208_200752.log.json) |
# Hybrid Task Cascade for Instance Segmentation

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the results in the CVPR 2019 paper for [Hybrid Task Cascade](https://arxiv.org/abs/1901.07518).

```latex
@inproceedings{chen2019hybrid,
  title={Hybrid task cascade for instance segmentation},
  author={Chen, Kai and Pang, Jiangmiao and Wang, Jiaqi and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Shi, Jianping and Ouyang, Wanli and Chen Change Loy and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Dataset

HTC requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN  | pytorch | 1x      | 8.2      | 5.8            | 42.3   | 37.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317_070435.log.json) |
| R-50-FPN  | pytorch | 20e     | 8.2      | -              | 43.3   | 38.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_r50_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319-fe28c577.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319_070313.log.json) |
| R-101-FPN | pytorch | 20e     | 10.2     | 5.5            | 44.8   | 39.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_r101_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r101_fpn_20e_coco/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_r101_fpn_20e_coco/htc_r101_fpn_20e_coco_20200317_153107.log.json) |
| X-101-32x4d-FPN | pytorch |20e| 11.4     | 5.0            | 46.1   | 40.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_x101_32x4d_fpn_16x1_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_32x4d_fpn_16x1_20e_coco/htc_x101_32x4d_fpn_16x1_20e_coco_20200318-de97ae01.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_32x4d_fpn_16x1_20e_coco/htc_x101_32x4d_fpn_16x1_20e_coco_20200318_034519.log.json) |
| X-101-64x4d-FPN | pytorch |20e| 14.5     | 4.4            | 47.0   | 41.4    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_16x1_20e_coco/htc_x101_64x4d_fpn_16x1_20e_coco_20200318_081711.log.json) |

- In the HTC paper and COCO 2018 Challenge, `score_thr` is set to 0.001 for both baselines and HTC.
- We use 8 GPUs with 2 images/GPU for R-50 and R-101 models, and 16 GPUs with 1 image/GPU for X-101 models.
  If you would like to train X-101 HTC with 8 GPUs, you need to change the lr from 0.02 to 0.01.

We also provide a powerful HTC with DCN and multi-scale training model. No testing augmentation is used.

| Backbone         | Style   | DCN   | training scales | Lr schd | box AP | mask AP | Config | Download |
|:----------------:|:-------:|:-----:|:---------------:|:-------:|:------:|:-------:|:------:|:--------:|
| X-101-64x4d-FPN  | pytorch | c3-c5 | 400~1400        | 20e     | 50.4   | 43.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312_203410.log.json) |
# Rethinking Classification and Localization for Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{wu2019rethinking,
    title={Rethinking Classification and Localization for Object Detection},
    author={Yue Wu and Yinpeng Chen and Lu Yuan and Zicheng Liu and Lijuan Wang and Hongzhi Li and Yun Fu},
    year={2019},
    eprint={1904.06493},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    | 6.8      | 9.5            | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/double_heads/dh_faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130-586b67df.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/double_heads/dh_faster_rcnn_r50_fpn_1x_coco/dh_faster_rcnn_r50_fpn_1x_coco_20200130_220238.log.json) |
# You Only Look One-level Feature

## Introduction

<!-- [ALGORITHM] -->

```
@inproceedings{chen2021you,
  title={You Only Look One-level Feature},
  author={Chen, Qiang and Wang, Yingming and Yang, Tong and Zhang, Xiangyu and Cheng, Jian and Sun, Jian},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## Results and Models

| Backbone  | Style   | Epoch | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50-C5     | caffe | Y | 1x      | 8.3      |   37.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py)       |[model](http://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolof/yolof_r50_c5_8x8_1x_coco/yolof_r50_c5_8x8_1x_coco_20210425_024427.log.json) |

**Note**:

1. We find that the performance is unstable and may fluctuate by about 0.3 mAP. mAP 37.4 ~ 37.7 is acceptable in YOLOF_R_50_C5_1x. Such fluctuation can also be found in the [original implementation](https://github.com/chensnathan/YOLOF).
2. In addition to instability issues, sometimes there are large loss fluctuations and NAN, so there may still be problems with this project, which will be improved subsequently.
# WIDER Face Dataset

<!-- [DATASET] -->

To use the WIDER Face dataset you need to download it
and extract to the `data/WIDERFace` folder. Annotation in the VOC format
can be found in this [repo](https://github.com/sovrasov/wider-face-pascal-voc-annotations.git).
You should move the annotation files from `WIDER_train_annotations` and `WIDER_val_annotations` folders
to the `Annotation` folders inside the corresponding directories `WIDER_train` and `WIDER_val`.
Also annotation lists `val.txt` and `train.txt` should be copied to `data/WIDERFace` from `WIDER_train_annotations` and `WIDER_val_annotations`.
The directory should be like this:

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── WIDERFace
│   │   ├── WIDER_train
│   |   │   ├──0--Parade
│   |   │   ├── ...
│   |   │   ├── Annotations
│   │   ├── WIDER_val
│   |   │   ├──0--Parade
│   |   │   ├── ...
│   |   │   ├── Annotations
│   │   ├── val.txt
│   │   ├── train.txt

```

After that you can train the SSD300 on WIDER by launching training with the `ssd300_wider_face.py` config or
create your own config based on the presented one.

```
@inproceedings{yang2016wider,
   Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
   Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   Title = {WIDER FACE: A Face Detection Benchmark},
   Year = {2016}
}
```
# AutoAssign: Differentiable Label Assignment for Dense Object Detection

## Introduction

<!-- [ALGORITHM] -->

```
@article{zhu2020autoassign,
  title={AutoAssign: Differentiable Label Assignment for Dense Object Detection},
  author={Zhu, Benjin and Wang, Jianfeng and Jiang, Zhengkai and Zong, Fuhang and Liu, Songtao and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2007.03496},
  year={2020}
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:------:|:------:|:--------:|
| R-50     | pytorch | 1x      | 4.08      |   40.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py)       |[model](https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/autoassign/auto_assign_r50_fpn_1x_coco/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.log.json) |

**Note**:

1. We find that the performance is unstable with 1x setting and may fluctuate by about 0.3 mAP. mAP 40.3 ~ 40.6 is acceptable. Such fluctuation can also be found in the original implementation.
2. You can get a more stable results ~ mAP 40.6 with a schedule total 13 epoch, and learning rate is divided by 10 at 10th and 13th epoch.
# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s

<!-- [ALGORITHM] -->

```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
```

A simple, fully convolutional model for real-time instance segmentation. This is the code for our paper:

- [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 <!-- - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218) -->

For a real-time demo, check out our ICCV video:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/0pMfmo8qfpQ/0.jpg)](https://www.youtube.com/watch?v=0pMfmo8qfpQ)

## Evaluation

Here are our YOLACT models along with their FPS on a Titan Xp and mAP on COCO's `val`:

| Image Size | GPU x BS | Backbone      | *FPS  | mAP  | Weights | Configs | Download |
|:----------:|:--------:|:-------------:|:-----:|:----:|:-------:|:------:|:--------:|
| 550        | 1x8      | Resnet50-FPN  | 42.5 | 29.0 | | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolact/yolact_r50_1x8_coco.py) |[model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco_20200908-f38d58df.pth) |
| 550        | 8x8      | Resnet50-FPN  | 42.5 | 28.4 | | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolact/yolact_r50_8x8_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r50_8x8_coco_20200908-ca34f5db.pth) |
| 550        | 1x8      | Resnet101-FPN | 33.5 | 30.4 | | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolact/yolact_r101_1x8_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco_20200908-4cbe9101.pth) |

*Note: The FPS is evaluated by the [original implementation](https://github.com/dbolya/yolact). When calculating FPS, only the model inference time is taken into account. Data loading and post-processing operations such as converting masks to RLE code, generating COCO JSON results, image rendering are not included.

## Training

All the aforementioned models are trained with a single GPU. It typically takes ~12GB VRAM when using resnet-101 as the backbone. If you want to try multiple GPUs training, you may have to modify the configuration files accordingly, such as adjusting the training schedule and freezing batch norm.

```Shell
# Trains using the resnet-101 backbone with a batch size of 8 on a single GPU.
./tools/dist_train.sh configs/yolact/yolact_r101.py 1
```

## Testing

Please refer to [mmdetection/docs/getting_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md#inference-with-pretrained-models).

## Citation

If you use YOLACT or this code base in your work, please cite

```latex
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

<!-- For YOLACT++, please cite

```latex
@misc{yolact-plus-arxiv2019,
  title         = {YOLACT++: Better Real-time Instance Segmentation},
  author        = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  year          = {2019},
  eprint        = {1912.06218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
``` -->
# Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{zhang2019bridging,
  title   =  {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author  =  {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  journal =  {arXiv preprint arXiv:1912.02424},
  year    =  {2019}
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | pytorch | 1x      | 3.7      | 19.7           |  39.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss/atss_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209_102539.log.json) |
| R-101     | pytorch | 1x      | 5.6      | 12.3           |  41.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/atss/atss_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r101_fpn_1x_coco/atss_r101_fpn_1x_20200825-dfcadd6f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r101_fpn_1x_coco/atss_r101_fpn_1x_20200825-dfcadd6f.log.json) |
# Scale-Aware Trident Networks for Object Detection

## Introduction

<!-- [ALGORITHM] -->

```
@InProceedings{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={The International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Results and models

We reports the test results using only one branch for inference.

|    Backbone     |  Style  | mstrain | Lr schd | Mem (GB) | Inf time (fps) | box AP | Download |
| :-------------: | :-----: | :-----: | :-----: | :------: | :------------: | :----: | :------: |
|    R-50         |  caffe  |    N    |   1x    |          |                | 37.7   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838-2ec0b530.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_1x_coco/tridentnet_r50_caffe_1x_coco_20201230_141838.log.json) |
|    R-50         |  caffe  |    Y    |   1x    |          |                | 37.6   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839-6ce55ccb.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_1x_coco/tridentnet_r50_caffe_mstrain_1x_coco_20201230_141839.log.json) |
|    R-50         |  caffe  |    Y    |   3x    |          |                | 40.3   |[model](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539-46d227ba.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco/tridentnet_r50_caffe_mstrain_3x_coco_20201130_100539.log.json) |

**Note**

Similar to [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/projects/TridentNet), we haven't implemented the Scale-aware Training Scheme in section 4.2 of the paper.
# Cityscapes Dataset

<!-- [DATASET] -->

```
@inproceedings{Cordts2016Cityscapes,
   title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
   author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
   booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   year={2016}
}
```

## Common settings

- All baselines were trained using 8 GPU with a batch size of 8 (1 images per GPU) using the [linear scaling rule](https://arxiv.org/abs/1706.02677) to scale the learning rate.
- All models were trained on `cityscapes_train`, and tested on `cityscapes_val`.
- 1x training schedule indicates 64 epochs which corresponds to slightly less than the 24k iterations reported in the original schedule from the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870)
- COCO pre-trained weights are used to initialize.
- A conversion [script](../../tools/dataset_converters/cityscapes.py) is provided to convert Cityscapes into COCO format. Please refer to [install.md](../../docs/1_exist_data_model.md#prepare-datasets) for details.
- `CityscapesDataset` implemented three evaluation methods. `bbox` and `segm` are standard COCO bbox/mask AP. `cityscapes` is the cityscapes dataset official evaluation, which may be slightly higher than COCO.

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Mem (GB) | Inf time (fps) | box AP | Config | Download   |
| :-------------: | :-----: | :-----: | :---:    | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 |   5.2    |       -        |  40.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502_114915.log.json) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Scale    | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------: | :------------: | :----: | :-----: | :------: | :------: |
|    R-50-FPN     | pytorch |   1x    | 800-1024 |   5.3    |       -        |  40.9  |  36.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733.log.json) |
# SSD: Single Shot MultiBox Detector

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
}
```

## Results and models

| Backbone | Size  | Style | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------: | :---: | :---: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|  VGG16   |  300  | caffe |  120e   |   10.2   |  43.7          |  25.6  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd300_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307-a92d2092.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20200307_174216.log.json) |
|  VGG16   |  512  | caffe |  120e   |   9.3    |  30.7          |  29.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/ssd/ssd512_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308_134447.log.json) |
# ResNeSt: Split-Attention Networks

## Introduction

[BACKBONE]

```latex
@article{zhang2020resnest,
title={ResNeSt: Split-Attention Networks},
author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
journal={arXiv preprint arXiv:2004.08955},
year={2020}
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|S-50-FPN       | pytorch |   1x   |   4.8  |   -           | 42.0 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20200926_125502-20289c16.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20200926_125502.log.json) |
|S-101-FPN       | pytorch |   1x   |   7.1  |   -           | 44.5 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/faster_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201006_021058-421517f1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/faster_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20201006_021058.log.json) |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|S-50-FPN       | pytorch |    1x   |   5.5  |      -         | 42.6 | 38.1 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20200926_125503-8a2c3d47.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20200926_125503.log.json) |
|S-101-FPN       | pytorch |    1x   |   7.8  |      -         | 45.2 | 40.2 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_215831-af60cdf9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20201005_215831.log.json) |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|S-50-FPN       | pytorch |   1x   |   -    |   -           |  44.5  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201122_213640-763cc7b5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20201005_113242.log.json) |
|S-101-FPN       | pytorch |   1x   |   8.4  |   -           |  46.8  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_rcnn_s101_fpn_syncbn-backbone+head_mstrain-range_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco_20201005_113242-b9459f8f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco/cascade_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain-range_1x_coco-20201122_213640.log.json) |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|S-50-FPN       | pytorch |    1x   |   -    |      -         | 45.4 | 39.5 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201122_104428-99eca4c7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s50_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20201122_104428.log.json) |
|S-101-FPN       | pytorch |    1x   |  10.5  |      -         | 47.7 | 41.4 |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco_20201005_113243-42607475.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/resnest/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco/cascade_mask_rcnn_s101_fpn_syncbn-backbone%2Bhead_mstrain_1x_coco-20201005_113243.log.json) |
# Deformable Convolutional Networks

## Introduction

<!-- [ALGORITHM] -->

```none
@inproceedings{dai2017deformable,
  title={Deformable Convolutional Networks},
  author={Dai, Jifeng and Qi, Haozhi and Xiong, Yuwen and Li, Yi and Zhang, Guodong and Hu, Han and Wei, Yichen},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
```

<!-- [ALGORITHM] -->

```
@article{zhu2018deformable,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1811.11168},
  year={2018}
}
```

## Results and Models

| Backbone         | Model        | Style   | Conv          | Pool   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:----------------:|:------------:|:-------:|:-------------:|:------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50-FPN         | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 4.0  | 17.8 | 41.3 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130_212941.log.json) |
| R-50-FPN         | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 4.1  | 17.6 | 41.4 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130-d099253b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200130_222144.log.json) |
| *R-50-FPN (dg=4) | Faster       | pytorch | mdconv(c3-c5) | -      | 1x      | 4.2  | 17.4 | 41.5 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130-01262257.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco/faster_rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco_20200130_222058.log.json) |
| R-50-FPN         | Faster       | pytorch | -             | dpool  | 1x      | 5.0  | 17.2 | 38.9 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_dpool_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307-90d3c01d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_dpool_1x_coco/faster_rcnn_r50_fpn_dpool_1x_coco_20200307_203250.log.json) |
| R-50-FPN         | Faster       | pytorch | -             | mdpool | 1x      | 5.8  | 16.6 | 38.7 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307-c0df27ff.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r50_fpn_mdpool_1x_coco/faster_rcnn_r50_fpn_mdpool_1x_coco_20200307_203304.log.json) |
| R-101-FPN        | Faster       | pytorch | dconv(c3-c5)  | -      | 1x      | 6.0  | 12.5 | 42.7 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-1377f13d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco/faster_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203_230019.log.json) |
| X-101-32x4d-FPN | Faster        | pytorch | dconv(c3-c5)  | -      | 1x      | 7.3  | 10.0  | 44.5 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203-4f85c69c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/faster_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco_20200203_001325.log.json) |
| R-50-FPN         | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5  | 15.4 | 41.8 | 37.4 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203_061339.log.json) |
| R-50-FPN         | Mask         | pytorch | mdconv(c3-c5) | -      | 1x      | 4.5  | 15.1 | 41.5 | 37.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203-ad97591f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_20200203_063443.log.json) |
| R-101-FPN        | Mask         | pytorch | dconv(c3-c5)  | -      | 1x      | 6.5  | 11.7  | 43.5 | 38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216_191601.log.json) |
| R-50-FPN         | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 4.5  | 14.6 | 43.8 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130_220843.log.json) |
| R-101-FPN        | Cascade      | pytorch | dconv(c3-c5)  | -      | 1x      | 6.4  | 11.0 | 45.0 |     | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203-3b2f0594.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200203_224829.log.json) |
| R-50-FPN         | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 6.0  | 10.0  | 44.4 | 38.6 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202-42e767a2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200202_010309.log.json) |
| R-101-FPN        | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 8.0  | 8.6  | 45.8 | 39.7 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204-df0c5f10.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200204_134006.log.json) |
| X-101-32x4d-FPN        | Cascade Mask | pytorch | dconv(c3-c5)  | -      | 1x      | 9.2 |   | 47.3 | 41.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-20200606_183737.log.json) |

**Notes:**

- `dconv` and `mdconv` denote (modulated) deformable convolution, `c3-c5` means adding dconv in resnet stage 3 to 5. `dpool` and `mdpool` denote (modulated) deformable roi pooling.
- The dcn ops are modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch, which should be more memory efficient and slightly faster.
- (*) For R-50-FPN (dg=4), dg is short for deformable_group. This model is trained and tested on Amazon EC2 p3dn.24xlarge instance.
- **Memory, Train/Inf time is outdated.**
# Region Proposal by Guided Anchoring

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the results in the CVPR 2019 paper for [Region Proposal by Guided Anchoring](https://arxiv.org/abs/1901.03278).

```latex
@inproceedings{wang2019region,
    title={Region Proposal by Guided Anchoring},
    author={Jiaqi Wang and Kai Chen and Shuo Yang and Chen Change Loy and Dahua Lin},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).

| Method |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | AR 1000 | Config | Download |
| :----: | :-------------: | :-----: | :-----: | :------: | :------------: | :-----: | :------: | :--------: |
| GA-RPN |    R-50-FPN     |  caffe  |   1x    |   5.3    |      15.8      |  68.4   |   [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco/ga_rpn_r50_caffe_fpn_1x_coco_20200531-899008a6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r50_caffe_fpn_1x_coco/ga_rpn_r50_caffe_fpn_1x_coco_20200531_011819.log.json)   |
| GA-RPN |    R-101-FPN    |  caffe  |   1x    |   7.3    |      13.0      |  69.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco/ga_rpn_r101_caffe_fpn_1x_coco_20200531-ca9ba8fb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_r101_caffe_fpn_1x_coco/ga_rpn_r101_caffe_fpn_1x_coco_20200531_011812.log.json) |
| GA-RPN | X-101-32x4d-FPN | pytorch |   1x    |   8.5    |      10.0      |  70.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220-c28d1b18.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_32x4d_fpn_1x_coco/ga_rpn_x101_32x4d_fpn_1x_coco_20200220_221326.log.json) |
| GA-RPN | X-101-64x4d-FPN | pytorch |   1x    |   7.1    |      7.5       |  71.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225-3c6e1aa2.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_rpn_x101_64x4d_fpn_1x_coco/ga_rpn_x101_64x4d_fpn_1x_coco_20200225_152704.log.json) |

|     Method     |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :------------: | :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
| GA-Faster RCNN |    R-50-FPN     |  caffe  |   1x    |   5.5    |                |  39.6  |          [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718-a11ccfe6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r50_caffe_fpn_1x_coco/ga_faster_r50_caffe_fpn_1x_coco_20200702_000718.log.json)           |
| GA-Faster RCNN |    R-101-FPN    |  caffe  |   1x    |   7.5    |                |  41.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco/ga_faster_r101_caffe_fpn_1x_coco_bbox_mAP-0.415_20200505_115528-fb82e499.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_r101_caffe_fpn_1x_coco/ga_faster_r101_caffe_fpn_1x_coco_20200505_115528.log.json) |
| GA-Faster RCNN | X-101-32x4d-FPN | pytorch |   1x    |   8.7    |      9.7       |  43.0  |            [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco/ga_faster_x101_32x4d_fpn_1x_coco_20200215-1ded9da3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_32x4d_fpn_1x_coco/ga_faster_x101_32x4d_fpn_1x_coco_20200215_184547.log.json)            |
| GA-Faster RCNN | X-101-64x4d-FPN | pytorch |   1x    |   11.8   |      7.3       |  43.9  |            [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco/ga_faster_x101_64x4d_fpn_1x_coco_20200215-0fa7bde7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_faster_x101_64x4d_fpn_1x_coco/ga_faster_x101_64x4d_fpn_1x_coco_20200215_104455.log.json)            |
|  GA-RetinaNet  |    R-50-FPN     |  caffe  |   1x    |   3.5    |      16.8      |  36.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco/ga_retinanet_r50_caffe_fpn_1x_coco_20201020-39581c6f.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x_coco/ga_retinanet_r50_caffe_fpn_1x_coco_20201020_225450.log.json)       |
|  GA-RetinaNet  |    R-101-FPN    |  caffe  |   1x    |   5.5    |      12.9      |  39.0  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco/ga_retinanet_r101_caffe_fpn_1x_coco_20200531-6266453c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_r101_caffe_fpn_1x_coco/ga_retinanet_r101_caffe_fpn_1x_coco_20200531_012847.log.json)      |
|  GA-RetinaNet  | X-101-32x4d-FPN | pytorch |   1x    |   6.9    |      10.6      |  40.5  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219-40c56caa.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x_coco/ga_retinanet_x101_32x4d_fpn_1x_coco_20200219_223025.log.json)      |
|  GA-RetinaNet  | X-101-64x4d-FPN | pytorch |   1x    |   9.9    |      7.7       |  41.3  |      [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226-ef9f7f1f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/guided_anchoring/ga_retinanet_x101_64x4d_fpn_1x_coco/ga_retinanet_x101_64x4d_fpn_1x_coco_20200226_221123.log.json)      |

- In the Guided Anchoring paper, `score_thr` is set to 0.001 in Fast/Faster RCNN and 0.05 in RetinaNet for both baselines and Guided Anchoring.

- Performance on COCO test-dev benchmark are shown as follows.

|     Method     | Backbone  | Style | Lr schd | Aug Train | Score thr |  AP   | AP_50 | AP_75 | AP_small | AP_medium | AP_large | Download |
| :------------: | :-------: | :---: | :-----: | :-------: | :-------: | :---: | :---: | :---: | :------: | :-------: | :------: | :------: |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.05    |       |       |       |          |           |          |          |
| GA-Faster RCNN | R-101-FPN | caffe |   1x    |     F     |   0.001   |       |       |       |          |           |          |          |
|  GA-RetinaNet  | R-101-FPN | caffe |   1x    |     F     |   0.05    |       |       |       |          |           |          |          |
|  GA-RetinaNet  | R-101-FPN | caffe |   2x    |     T     |   0.05    |       |       |       |          |           |          |          |
# Deformable DETR

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for Deformable DETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).

```
@inproceedings{
zhu2021deformable,
title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```

## Results and Models

| Backbone | Model | Lr schd  | box AP | Config | Download |
|:------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | Deformable DETR  |50e  | 44.5 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.log.json) |
| R-50 | + iterative bounding box refinement  |50e  | 46.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_refine_r50_16x2_50e_coco/deformable_detr_refine_r50_16x2_50e_coco_20210419_220503-5f5dff21.log.json) |
| R-50 | ++ two-stage Deformable DETR  |50e  | 46.8 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.log.json) |

# NOTE

1. All models are trained with batch size 32.
2. The performance is unstable. `Deformable DETR` and `iterative bounding box refinement` may fluctuate about 0.3 mAP. `two-stage Deformable DETR` may fluctuate about 0.2 mAP.
# PointRend

## Introduction

<!-- [ALGORITHM] -->

```latex
@InProceedings{kirillov2019pointrend,
  title={{PointRend}: Image Segmentation as Rendering},
  author={Alexander Kirillov and Yuxin Wu and Kaiming He and Ross Girshick},
  journal={ArXiv:1912.08193},
  year={2019}
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    | 4.6      |                | 38.4   | 36.3    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco_20200612_161407.log.json) |
|    R-50-FPN     |  caffe  |   3x    | 4.6      |                | 41.0   | 38.0    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco_20200614_002632.log.json) |

Note: All models are trained with multi-scale, the input image shorter side is randomly scaled to one of (640, 672, 704, 736, 768, 800).
# YOLOv3

## Introduction

<!-- [ALGORITHM] -->

```latex
@misc{redmon2018yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Joseph Redmon and Ali Farhadi},
    year={2018},
    eprint={1804.02767},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and Models

|    Backbone     |  Scale  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|   DarkNet-53    |   320   |   273e  |   2.7    |      63.9      |  27.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_320_273e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-20200819_172101.log.json) |
|   DarkNet-53    |   416   |   273e  |   3.8    |      61.2      |  30.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_mstrain-416_273e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-20200819_173424.log.json) |
|   DarkNet-53    |   608   |   273e  |   7.1    |      48.1      |  33.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-139f5633.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco-20200819_170820.log.json) |

## Credit

This implementation originates from the project of Haoyu Wu(@wuhy08) at Western Digital.
# Legacy Configs in MMDetection V1.x

<!-- [OTHERS] -->

Configs in this directory implement the legacy configs used by MMDetection V1.x and its model zoos.

To help users convert their models from V1.x to MMDetection V2.0, we provide v1.x configs to inference the converted v1.x models.
Due to the BC-breaking changes in MMDetection V2.0 from MMDetection V1.x, running inference with the same model weights in these two version will produce different results. The difference will cause within 1% AP absolute difference as can be found in the following table.

## Usage

To upgrade the model version, the users need to do the following steps.

### 1. Convert model weights

There are three main difference in the model weights between V1.x and V2.0 codebases.

1. Since the class order in all the detector's classification branch is reordered, all the legacy model weights need to go through the conversion process.
2. The regression and segmentation head no longer contain the background channel. Weights in these background channels should be removed to fix in the current codebase.
3. For two-stage detectors, their wegihts need to be upgraded since MMDetection V2.0 refactors all the two-stage detectors with `RoIHead`.

The users can do the same modification as mentioned above for the self-implemented
detectors. We provide a scripts `tools/model_converters/upgrade_model_version.py` to convert the model weights in the V1.x model zoo.

```bash
python tools/model_converters/upgrade_model_version.py ${OLD_MODEL_PATH} ${NEW_MODEL_PATH} --num-classes ${NUM_CLASSES}

```

- OLD_MODEL_PATH: the path to load the model weights in 1.x version.
- NEW_MODEL_PATH: the path to save the converted model weights in 2.0 version.
- NUM_CLASSES: number of classes of the original model weights. Usually it is 81 for COCO dataset, 21 for VOC dataset.
  The number of classes in V2.0 models should be equal to that in V1.x models - 1.

### 2. Use configs with legacy settings

After converting the model weights, checkout to the v1.2 release to find the corresponding config file that uses the legacy settings.
The V1.x models usually need these three legacy modules: `LegacyAnchorGenerator`, `LegacyDeltaXYWHBBoxCoder`, and `RoIAlign(align=False)`.
For models using ResNet Caffe backbones, they also need to change the pretrain name and the corresponding `img_norm_cfg`.
An example is in [`retinanet_r50_caffe_fpn_1x_coco_v1.py`](retinanet_r50_caffe_fpn_1x_coco_v1.py)
Then use the config to test the model weights. For most models, the obtained results should be close to that in V1.x.
We provide configs of some common structures in this directory.

## Performance

The performance change after converting the models in this directory are listed as the following.
|    Method    |  Style  | Lr schd | V1.x box AP | V1.x mask AP | V2.0 box AP | V2.0 mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------:| :-----: |:------:| :-----: | :-------: |:------------------------------------------------------------------------------------------------------------------------------: |
| Mask R-CNN R-50-FPN     | pytorch |   1x    |  37.3  |  34.2   | 36.8 | 33.9 | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/legacy_1.x/mask_rcnn_r50_fpn_1x_coco_v1.py) | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth)|
| RetinaNet R-50-FPN |  caffe  |   1x    |  35.8  | - | 35.4 | - | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/legacy_1.x/retinanet_r50_caffe_1x_coco_v1.py) |
| RetinaNet R-50-FPN | pytorch |   1x |  35.6 |-|35.2|   -| [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/legacy_1.x/retinanet_r50_fpn_1x_coco_v1.py) | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth)     |
| Cascade Mask R-CNN R-50-FPN | pytorch |   1x    |  41.2  |  35.7   |40.8| 35.6|  [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/legacy_1.x/cascade_mask_rcnn_r50_fpn_1x_coco_v1.py) |     [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth)     |
| SSD300-VGG16 | caffe |  120e   | 25.7  |-|25.4|-|  [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/legacy_1.x/ssd300_coco_v1.py) | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth) |
# SCNet

## Introduction

<!-- [ALGORITHM] -->

We provide the code for reproducing experiment results of [SCNet](https://arxiv.org/abs/2012.10150).

```
@inproceedings{vu2019cascade,
  title={SCNet: Training Inference Sample Consistency for Instance Segmentation},
  author={Vu, Thang and Haeyong, Kang and Yoo, Chang D},
  booktitle={AAAI},
  year={2021}
}
```

## Dataset

SCNet requires COCO and [COCO-stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) dataset for training. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
|   |   ├── stuffthingmaps
```

## Results and Models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

|     Backbone    |  Style  | Lr schd | Mem (GB) | Inf speed (fps) | box AP | mask AP | TTA box AP | TTA mask AP | Config |   Download   |
|:---------------:|:-------:|:-------:|:--------:|:---------------:|:------:|:-------:|:----------:|:-----------:|:------:|:------------:|
|     R-50-FPN    | pytorch |    1x   |    7.0   |       6.2       |  43.5  |   39.2  |    44.8    |     40.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1K5_8-P0EC43WZFtoO3q9_JE-df8pEc7J/view?usp=sharing) \| [log](https://drive.google.com/file/d/1ZFS6QhFfxlOnDYPiGpSDP_Fzgb7iDGN3/view?usp=sharing) |
|     R-50-FPN    | pytorch |   20e   |    7.0   |       6.2       |  44.5  |   40.0  |    45.8    |     41.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r50_fpn_20e_coco.py) | [model](https://drive.google.com/file/d/15VGLCt5-IO5TbzB4Kw6ZyoF6QH0Q511A/view?usp=sharing) \| [log](https://drive.google.com/file/d/1-LnkOXN8n5ojQW34H0qZ625cgrnWpqSX/view?usp=sharing) |
|    R-101-FPN    | pytorch |   20e   |    8.9   |       5.8       |  45.8  |   40.9  |    47.3    |     42.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r101_fpn_20e_coco.py) | [model](https://drive.google.com/file/d/1aeCGHsOBdfIqVBnBPp0JUE_RSIau3583/view?usp=sharing) \| [log](https://drive.google.com/file/d/1iRx-9GRgTaIDsz-we3DGwFVH22nbvCLa/view?usp=sharing) |
| X-101-64x4d-FPN | pytorch |   20e   |   13.2   |       4.9       |  47.5  |   42.3  |    48.9    |     44.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py) | [model](https://drive.google.com/file/d/1YjgutUKz4TTPpqSWGKUTkZJ8_X-kyCfY/view?usp=sharing) \| [log](https://drive.google.com/file/d/1OsfQJ8gwtqIQ61k358yxY21sCvbUcRjs/view?usp=sharing) |

### Notes

- Training hyper-parameters are identical to those of [HTC](https://github.com/open-mmlab/mmdetection/tree/master/configs/htc).
- TTA means Test Time Augmentation, which applies horizonal flip and multi-scale testing. Refer to [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/scnet/scnet_r50_fpn_1x_coco.py).
# DETR

## Introduction

<!-- [ALGORITHM] -->

We provide the config files for DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

```BibTeX
@inproceedings{detr,
  author    = {Nicolas Carion and
               Francisco Massa and
               Gabriel Synnaeve and
               Nicolas Usunier and
               Alexander Kirillov and
               Sergey Zagoruyko},
  title     = {End-to-End Object Detection with Transformers},
  booktitle = {ECCV},
  year      = {2020}
}
```

## Results and Models

| Backbone | Model | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------:|:--------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | DETR  |150e |7.9|  | 40.1 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/detr/detr_r50_8x2_150e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835.log.json) |
# Grid R-CNN

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{lu2019grid,
  title={Grid r-cnn},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@article{lu2019grid,
  title={Grid R-CNN Plus: Faster and Better},
  author={Lu, Xin and Li, Buyu and Yue, Yuxin and Li, Quanquan and Yan, Junjie},
  journal={arXiv preprint arXiv:1906.05688},
  year={2019}
}
```

## Results and Models

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50        | 2x      | 5.1      | 15.0           | 40.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130-6cca8223.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r50_fpn_gn-head_2x_coco/grid_rcnn_r50_fpn_gn-head_2x_coco_20200130_221140.log.json) |
| R-101       | 2x      | 7.0      | 12.6           | 41.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309-d6eca030.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco/grid_rcnn_r101_fpn_gn-head_2x_coco_20200309_164224.log.json) |
| X-101-32x4d | 2x      | 8.3      | 10.8           | 42.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130-d8f0e3ff.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_32x4d_fpn_gn-head_2x_coco_20200130_215413.log.json) |
| X-101-64x4d | 2x      | 11.3     | 7.7            | 43.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204-ec76a754.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/grid_rcnn/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco/grid_rcnn_x101_64x4d_fpn_gn-head_2x_coco_20200204_080641.log.json) |

**Notes:**

- All models are trained with 8 GPUs instead of 32 GPUs in the original paper.
- The warming up lasts for 1 epoch and `2x` here indicates 25 epochs.
# An Empirical Study of Spatial Attention Mechanisms in Deep Networks

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{zhu2019empirical,
  title={An Empirical Study of Spatial Attention Mechanisms in Deep Networks},
  author={Zhu, Xizhou and Cheng, Dazhi and Zhang, Zheng and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1904.05873},
  year={2019}
}
```

## Results and Models

| Backbone  | Attention Component | DCN  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------------------:|:----:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | 1111                | N    | 1x      | 8.0      | 13.8            | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130-403cccba.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_1x_coco/faster_rcnn_r50_fpn_attention_1111_1x_coco_20200130_210344.log.json) |
| R-50      | 0010                | N    | 1x      | 4.2      | 18.4           | 39.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130-7cb0c14d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco/faster_rcnn_r50_fpn_attention_0010_1x_coco_20200130_210125.log.json) |
| R-50      | 1111                | Y    | 1x      | 8.0      | 12.7            | 42.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130-8b2523a6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco/faster_rcnn_r50_fpn_attention_1111_dcn_1x_coco_20200130_204442.log.json) |
| R-50      | 0010                | Y    | 1x      | 4.2      | 17.1           | 42.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/empirical_attention/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130_210410.log.json) |
# Sparse R-CNN: End-to-End Object Detection with Learnable Proposals

## Introduction

<!-- [ALGORITHM] -->

```
@article{peize2020sparse,
  title   =  {{SparseR-CNN}: End-to-End Object Detection with Learnable Proposals},
  author  =  {Peize Sun and Rufeng Zhang and Yi Jiang and Tao Kong and Chenfeng Xu and Wei Zhan and Masayoshi Tomizuka and Lei Li and Zehuan Yuan and Changhu Wang and Ping Luo},
  journal =  {arXiv preprint arXiv:2011.12450},
  year    =  {2020}
}
```

## Results and Models

| Model        | Backbone  | Style   | Lr schd | Number of Proposals |Multi-Scale| RandomCrop  | box AP  | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:-------:            |:-------: |:---------:|:------:|:------:|:--------:|
| Sparse R-CNN | R-50-FPN  | pytorch | 1x      |   100               | False     |  False     |  37.9  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.log.json) |
| Sparse R-CNN | R-50-FPN  | pytorch | 3x      |   100               | True     |   False     |  42.8  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_mstrain_480-800_3x_coco_20201218_154234-7bc5c054.log.json) |
| Sparse R-CNN | R-50-FPN  | pytorch | 3x      |   300               | True      |  True      |  45.0  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.log.json) |
| Sparse R-CNN | R-101-FPN | pytorch | 3x      |   100               | True      |  False     |  44.2  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco_20201223_121552-6c46c9d6.log.json) |
| Sparse R-CNN | R-101-FPN | pytorch | 3x      |   300               | True      |  True      |  46.2  |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_023452-c23c3564.log.json) |

### Notes

We observe about 0.3 AP noise especially when using ResNet-101 as the backbone.
# Designing Network Design Spaces

## Introduction

[BACKBONE]

We implement RegNetX and RegNetY models in detection systems and provide their first results on Mask R-CNN, Faster R-CNN and RetinaNet.

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).

```latex
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Usage

To use a regnet model, there are two steps to do:

1. Convert the model to ResNet-style supported by MMDetection
2. Modify backbone and neck in config accordingly

### Convert model

We already prepare models of FLOPs from 400M to 12G in our model zoo.

For more general usage, we also provide script `regnet2mmdet.py` in the tools directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
ResNet-style checkpoints used in MMDetection.

```bash
python -u tools/model_converters/regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

### Modify config

The users can modify the config's `depth` of backbone and corresponding keys in `arch` according to the configs in the [pycls model zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).
The parameter `in_channels` in FPN can be found in the Figure 15 & 16 of the paper (`wi` in the legend).
This directory already provides some configs with their performance, using RegNetX from 800MF to 12GF level.
For other pre-trained models or self-implemented regnet models, the users are responsible to check these parameters by themselves.

**Note**: Although Fig. 15 & 16 also provide `w0`, `wa`, `wm`, `group_w`, and `bot_mul` for `arch`, they are quantized thus inaccurate, using them sometimes produces different backbone that does not match the key in the pre-trained model.

## Results

### Mask R-CNN

|   Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    [R-50-FPN](../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)| pytorch |   1x    | 4.4      | 12.0           | 38.2   | 34.7    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205_050542.log.json) |
|[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py)| pytorch |   1x    |5.0 ||40.3|36.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141-2a9d1814.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_1x_coco_20200520_163141.log.json)   |
|[RegNetX-4.0GF-FPN](./mask_rcnn_regnetx-4GF_fpn_1x_coco.py)| pytorch |   1x    |5.5||41.5|37.4|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217-32e9c92d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-4GF_fpn_1x_coco/mask_rcnn_regnetx-4GF_fpn_1x_coco_20200517_180217.log.json)   |
|    [R-101-FPN](../mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py)| pytorch |   1x    | 6.4      | 10.3           | 40.0   | 36.1    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_1x_coco/mask_rcnn_r101_fpn_1x_coco_20200204_144809.log.json) |
|[RegNetX-6.4GF-FPN](./mask_rcnn_regnetx-6.4GF_fpn_1x_coco.py)| pytorch |   1x    |6.1 ||41.0|37.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-6.4GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-6.4GF_fpn_1x_coco/mask_rcnn_regnetx-6.4GF_fpn_1x_coco_20200517_180439-3a7aae83.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-6.4GF_fpn_1x_coco/mask_rcnn_regnetx-6.4GF_fpn_1x_coco_20200517_180439.log.json)   |
| [X-101-32x4d-FPN](../mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py) | pytorch |   1x    | 7.6      | 9.4            | 41.9   | 37.5    |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205-478d0b67.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco/mask_rcnn_x101_32x4d_fpn_1x_coco_20200205_034906.log.json) |
|[RegNetX-8.0GF-FPN](./mask_rcnn_regnetx-8GF_fpn_1x_coco.py)| pytorch |   1x    |6.4 ||41.7|37.5|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515-09daa87e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-8GF_fpn_1x_coco/mask_rcnn_regnetx-8GF_fpn_1x_coco_20200517_180515.log.json)   |
|[RegNetX-12GF-FPN](./mask_rcnn_regnetx-12GF_fpn_1x_coco.py)| pytorch |   1x    |7.4 ||42.2|38|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552-b538bd8b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-12GF_fpn_1x_coco/mask_rcnn_regnetx-12GF_fpn_1x_coco_20200517_180552.log.json) |
|[RegNetX-3.2GF-FPN-DCN-C3-C5](./mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco.py)| pytorch |   1x    |5.0 ||40.3|36.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726-75f40794.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco/mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco_20200520_172726.log.json)   |

### Faster R-CNN

|   Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    [R-50-FPN](../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py)| pytorch |   1x    | 4.0      | 18.2           | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py)| pytorch |   1x    | 4.5||39.9|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_1x_coco/faster_rcnn_regnetx-3.2GF_fpn_1x_coco_20200517_175927-126fd9bf.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_1x_coco/faster_rcnn_regnetx-3.2GF_fpn_1x_coco_20200517_175927.log.json)   |
|[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3.2GF_fpn_2x_coco.py)| pytorch |   2x    | 4.5||41.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-3.2GF_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_2x_coco/faster_rcnn_regnetx-3.2GF_fpn_2x_coco_20200520_223955-e2081918.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_2x_coco/faster_rcnn_regnetx-3.2GF_fpn_2x_coco_20200520_223955.log.json)   |

### RetinaNet

|  Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :---------: | :-----: | :-----: | :------: | :------------: | :----: | :------: |  :--------: |
|    [R-50-FPN](../retinanet/retinanet_r50_fpn_1x_coco.py)     | pytorch |   1x    |   3.8    |      16.6      |  36.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130_002941.log.json) |
|[RegNetX-800MF-FPN](./retinanet_regnetx-800MF_fpn_1x_coco.py)| pytorch |   1x    |2.5||35.6|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/retinanet_regnetx-800MF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-800MF_fpn_1x_coco/retinanet_regnetx-800MF_fpn_1x_coco_20200517_191403-f6f91d10.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-800MF_fpn_1x_coco/retinanet_regnetx-800MF_fpn_1x_coco_20200517_191403.log.json)   |
|[RegNetX-1.6GF-FPN](./retinanet_regnetx-1.6GF_fpn_1x_coco.py)| pytorch |   1x    |3.3||37.3|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco/retinanet_regnetx-1.6GF_fpn_1x_coco_20200517_191403-37009a9d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-1.6GF_fpn_1x_coco/retinanet_regnetx-1.6GF_fpn_1x_coco_20200517_191403.log.json)   |
|[RegNetX-3.2GF-FPN](./retinanet_regnetx-3.2GF_fpn_1x_coco.py)| pytorch |   1x    |4.2 ||39.1|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco/retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141-cb1509e8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/retinanet_regnetx-3.2GF_fpn_1x_coco/retinanet_regnetx-3.2GF_fpn_1x_coco_20200520_163141.log.json)   |

### Pre-trained models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Method   |    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-----: | :-----: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: |  :--------: |
|Faster RCNN |[RegNetX-3.2GF-FPN](./faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |5.0 ||42.2|-|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200520_224253-bf85ae3e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/faster_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200520_224253.log.json)   |
|Mask RCNN |[RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py)| pytorch |   3x    |5.0 ||43.1|38.7|[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221-99879813.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/regnet/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco/mask_rcnn_regnetx-3.2GF_fpn_mstrain_3x_coco_20200521_202221.log.json)   |

### Notice

1. The models are trained using a different weight decay, i.e., `weight_decay=5e-5` according to the setting in ImageNet training. This brings improvement of at least 0.7 AP absolute but does not improve the model using ResNet-50.
2. RetinaNets using RegNets are trained with learning rate 0.02 with gradient clip. We find that using learning rate 0.02 could improve the results by at least 0.7 AP absolute and gradient clip is necessary to stabilize the training. However, this does not improve the performance of ResNet-50-FPN RetinaNet.
# Fast R-CNN

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{girshick2015fast,
  title={Fast r-cnn},
  author={Girshick, Ross},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2015}
}
```

## Results and models
# Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training

## Introduction

<!-- [ALGORITHM] -->

```
@article{DynamicRCNN,
    author = {Hongkai Zhang and Hong Chang and Bingpeng Ma and Naiyan Wang and Xilin Chen},
    title = {Dynamic {R-CNN}: Towards High Quality Object Detection via Dynamic Training},
    journal = {arXiv preprint arXiv:2004.06002},
    year = {2020}
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | pytorch | 1x      | 3.8      |                |  38.9  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x-62a3f276.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x/dynamic_rcnn_r50_fpn_1x_20200618_095048.log.json) |
# NAS-FCOS: Fast Neural Architecture Search for Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{wang2019fcos,
  title={Nas-fcos: Fast neural architecture search for object detection},
  author={Wang, Ning and Gao, Yang and Chen, Hao and Wang, Peng and Tian, Zhi and Shen, Chunhua},
  journal={arXiv preprint arXiv:1906.04423},
  year={2019}
}
```

## Results and Models

| Head      | Backbone  | Style   | GN-head | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:---------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| NAS-FCOSHead | R-50   | caffe   | Y       | 1x      |          |                | 39.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520-1bdba3ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200520.log.json) |
| FCOSHead  | R-50      | caffe   | Y       | 1x      |          |                | 38.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521-7fdcbce0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco_20200521.log.json) |

**Notes:**

- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU.
# Localization Distillation for Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@Article{zheng2021LD,
  title={Localization Distillation for Object Detection},
  author= {Zhaohui Zheng, Rongguang Ye, Ping Wang, Jun Wang, Dongwei Ren, Wangmeng Zuo},
  journal={arXiv:2102.12252},
  year={2021}
}
```

### GFocalV1 with LD

|  Teacher  | Student | Training schedule | Mini-batch size | AP (val) | AP50 (val) | AP75 (val) | Config |
| :-------: | :-----: | :---------------: | :-------------: | :------: | :--------: | :--------: |  :--------------: |
|    --     |  R-18   |        1x         |        6        |   35.8   |    53.1    |    38.2    |          |
|   R-101   |  R-18   |        1x         |        6        |   36.5   |    52.9    |    39.3    |   [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py)          |
|    --     |  R-34   |        1x         |        6        |   38.9   |    56.6    |    42.2    |          |
|   R-101   |  R-34   |        1x         |        6        |   39.8   |    56.6    |    43.1    |     [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r34_gflv1_r101_fpn_coco_1x.py)        |
|    --     |  R-50   |        1x         |        6        |   40.1   |    58.2    |    43.1    |            |
|   R-101   |  R-50   |        1x         |        6        |   41.1   |    58.7    |    44.9    |    [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py)        |
|    --     |  R-101  |        2x         |        6        |   44.6   |    62.9    |    48.4    |           |
| R-101-DCN |  R-101  |        2x         |        6        |   45.4   |    63.1    |    49.5    | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/ld/ld_r101_gflv1_r101dcn_fpn_coco_1x.py)           |

## Note

- Meaning of Config name: ld_r18(student model)_gflv1(based on gflv1)_r101(teacher model)_fpn(neck)_coco(dataset)_1x(12 epoch).py
# NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{ghiasi2019fpn,
  title={Nas-fpn: Learning scalable feature pyramid architecture for object detection},
  author={Ghiasi, Golnaz and Lin, Tsung-Yi and Le, Quoc V},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7036--7045},
  year={2019}
}
```

## Results and Models

We benchmark the new training schedule (crop training, large batch, unfrozen BN, 50 epochs) introduced in NAS-FPN. RetinaNet is used in the paper.

| Backbone    | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:-----------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50-FPN    | 50e     | 12.9     | 22.9           | 37.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco-9b953d76.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_fpn_crop640_50e_coco/retinanet_r50_fpn_crop640_50e_coco_20200529_095329.log.json) |
| R-50-NASFPN | 50e     | 13.2     | 23.0           | 40.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco-0ad1f644.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco/retinanet_r50_nasfpn_crop640_50e_coco_20200528_230008.log.json) |

**Note**: We find that it is unstable to train NAS-FPN and there is a small chance that results can be 3% mAP lower.
# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the object detection results in the paper [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388)

```latex
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
```

## Results and Models

| Backbone          | Style   | Lr schd | Multi-scale Training| Inf time (fps) | box AP | Config | Download |
|:-----------------:|:-------:|:-------:|:-------------------:|:--------------:|:------:|:------:|:--------:|
| R-50              | pytorch | 1x      | No                  | 19.5           | 40.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244.log.json) |
| R-50              | pytorch | 2x      | Yes                 | 19.5           | 42.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r50_fpn_mstrain_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_mstrain_2x_coco/gfl_r50_fpn_mstrain_2x_coco_20200629_213802.log.json) |
| R-101             | pytorch | 2x      | Yes                 | 14.7           | 44.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r101_fpn_mstrain_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126.log.json) |
| R-101-dcnv2       | pytorch | 2x      | Yes                 | 12.9           | 47.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002.log.json) |
| X-101-32x4d       | pytorch | 2x      | Yes                 | 12.1           | 45.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002-50c1ffdb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_mstrain_2x_coco/gfl_x101_32x4d_fpn_mstrain_2x_coco_20200630_102002.log.json) |
| X-101-32x4d-dcnv2 | pytorch | 2x      | Yes                 | 10.7           | 48.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002.log.json) |

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`dcnv2` denotes deformable convolutional networks v2.* \
[4] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.*
# CornerNet

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{law2018cornernet,
  title={Cornernet: Detecting objects as paired keypoints},
  author={Law, Hei and Deng, Jia},
  booktitle={15th European Conference on Computer Vision, ECCV 2018},
  pages={765--781},
  year={2018},
  organization={Springer Verlag}
}
```

## Results and models

| Backbone        | Batch Size | Step/Total Epochs | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :--------: |:----------------: | :------: | :------------: | :----: | :------: | :--------: |
| HourglassNet-104 | [10 x 5](./cornernet_hourglass104_mstest_10x5_210e_coco.py) | 180/210 | 13.9 | 4.2 | 41.2 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco/cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720-5fefbf1c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco/cornernet_hourglass104_mstest_10x5_210e_coco_20200824_185720.log.json) |
| HourglassNet-104 | [8 x 6](./cornernet_hourglass104_mstest_8x6_210e_coco.py) | 180/210 | 15.9 | 4.2 | 41.2 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618.log.json) |
| HourglassNet-104 | [32 x 3](./cornernet_hourglass104_mstest_32x3_210e_coco.py) | 180/210 | 9.5 | 3.9 | 40.4 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco/cornernet_hourglass104_mstest_32x3_210e_coco_20200819_203110-1efaea91.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco/cornernet_hourglass104_mstest_32x3_210e_coco_20200819_203110.log.json) |

Note:

- TTA setting is single-scale and `flip=True`.
- Experiments with `images_per_gpu=6` are conducted on Tesla V100-SXM2-32GB, `images_per_gpu=3` are conducted on GeForce GTX 1080 Ti.
- Here are the descriptions of each experiment setting:
  - 10 x 5: 10 GPUs with 5 images per gpu. This is the same setting as that reported in the original paper.
  - 8 x 6: 8 GPUs with 6 images per gpu. The total batchsize is similar to paper and only need 1 node to train.
  - 32 x 3: 32 GPUs with 3 images per gpu. The default setting for 1080TI and need 4 nodes to train.
# VarifocalNet: An IoU-aware Dense Object Detector

## Introduction

<!-- [ALGORITHM] -->

**VarifocalNet (VFNet)** learns to predict the IoU-aware classification score which mixes the object presence confidence and localization accuracy together as the detection score for a bounding box. The learning is supervised by the proposed Varifocal Loss (VFL), based on a new star-shaped bounding box feature representation (the features at nine yellow sampling points). Given the new representation, the object localization accuracy is further improved by refining the initially regressed bounding box. The full paper is available at: [https://arxiv.org/abs/2008.13367](https://arxiv.org/abs/2008.13367).

<div align="center">
  <img src="https://user-images.githubusercontent.com/9102141/97464778-4b9ab000-197c-11eb-9283-ab2907ee0252.png" width="600px" />
  <p>Learning to Predict the IoU-aware Classification Score.</p>
</div>

## Citing VarifocalNet

```latex
@article{zhang2020varifocalnet,
  title={VarifocalNet: An IoU-aware Dense Object Detector},
  author={Zhang, Haoyang and Wang, Ying and Dayoub, Feras and S{\"u}nderhauf, Niko},
  journal={arXiv preprint arXiv:2008.13367},
  year={2020}
}
```

## Results and Models

| Backbone     | Style     | DCN     | MS train | Lr schd |Inf time (fps) | box AP (val) | box AP (test-dev) | Config | Download |
|:------------:|:---------:|:-------:|:--------:|:-------:|:-------------:|:------------:|:-----------------:|:------:|:--------:|
| R-50         | pytorch   | N       | N        | 1x      | -          | 41.6         | 41.6              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_1x_coco.py) |  [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_1x_coco/vfnet_r50_fpn_1x_coco.json)|
| R-50         | pytorch   | N       | Y        | 2x      | -          | 44.5         | 44.8              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_mstrain_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco_20201027-7cc75bd2.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mstrain_2x_coco/vfnet_r50_fpn_mstrain_2x_coco.json)|
| R-50         | pytorch   | Y       | Y        | 2x      | -          | 47.8         | 48.0              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|
| R-101        | pytorch   | N       | N        | 1x      | -          | 43.0         | 43.6              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r101_fpn_1x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco_20201027pth-c831ece7.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_1x_coco/vfnet_r101_fpn_1x_coco.json)|
| R-101        | pytorch   | N       | Y        | 2x      | -          | 46.2         | 46.7              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r101_fpn_mstrain_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco_20201027pth-4a5d53f1.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mstrain_2x_coco/vfnet_r101_fpn_mstrain_2x_coco.json)|
| R-101        | pytorch   | Y       | Y        | 2x      | -          | 49.0         | 49.2              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-7729adb5.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|
| X-101-32x4d  | pytorch   | Y       | Y        | 2x      | -          | 49.7         | 50.0              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-d300a6fc.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_32x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|
| X-101-64x4d  | pytorch   | Y       | Y        | 2x      |  -         | 50.4         | 50.8              | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco.json)|

**Notes:**

- The MS-train scale range is 1333x[480:960] (`range` mode) and the inference scale keeps 1333x800.
- DCN means using `DCNv2` in both backbone and head.
- Inference time will be updated soon.
- More results and pre-trained models can be found in [VarifocalNet-Github](https://github.com/hyz-xmaster/VarifocalNet)
# Side-Aware Boundary Localization for More Precise Object Detection

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the object detection results in the ECCV 2020 Spotlight paper for [Side-Aware Boundary Localization for More Precise Object Detection](https://arxiv.org/abs/1912.04260).

```latex
@inproceedings{Wang_2020_ECCV,
    title = {Side-Aware Boundary Localization for More Precise Object Detection},
    author = {Jiaqi Wang and Wenwei Zhang and Yuhang Cao and Kai Chen and Jiangmiao Pang and Tao Gong and Jianping Shi and Chen Change Loy and Dahua Lin},
    booktitle = {ECCV},
    year = {2020}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table. (results on test-dev are usually slightly higher than val).
Single-scale testing (1333x800) is adopted in all results.

|       Method       | Backbone  | Lr schd | ms-train | box AP |                                                       Config                                                       |                                                                                                                                   Download                                                                                                                                    |
| :----------------: | :-------: | :-----: | :------: | :----: | :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SABL Faster R-CNN  | R-50-FPN  |   1x    |    N     |  39.9  |  [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_faster_rcnn_r50_fpn_1x_coco.py)  |    [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/sabl_faster_rcnn_r50_fpn_1x_coco-e867595b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r50_fpn_1x_coco/20200830_130324.log.json)    |
| SABL Faster R-CNN  | R-101-FPN |   1x    |    N     |  41.7  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_faster_rcnn_r101_fpn_1x_coco.py)  |  [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/sabl_faster_rcnn_r101_fpn_1x_coco-f804c6c1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_faster_rcnn_r101_fpn_1x_coco/20200830_183949.log.json)   |
| SABL Cascade R-CNN | R-50-FPN  |   1x    |    N     |  41.6  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco.py)  |  [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/sabl_cascade_rcnn_r50_fpn_1x_coco-e1748e5e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r50_fpn_1x_coco/20200831_033726.log.json)   |
| SABL Cascade R-CNN | R-101-FPN |   1x    |    N     |  43.0  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/sabl_cascade_rcnn_r101_fpn_1x_coco-2b83e87c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_cascade_rcnn_r101_fpn_1x_coco/20200831_141745.log.json) |

|     Method     | Backbone  |  GN   | Lr schd |  ms-train   | box AP |                                                            Config                                                             |                                                                                                                                                    Download                                                                                                                                                    |
| :------------: | :-------: | :---: | :-----: | :---------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| SABL RetinaNet | R-50-FPN  |   N   |   1x    |      N      |  37.7  |        [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r50_fpn_1x_coco.py)         |                       [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/sabl_retinanet_r50_fpn_1x_coco-6c54fd4f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_1x_coco/20200830_053451.log.json)                        |
| SABL RetinaNet | R-50-FPN  |   Y   |   1x    |      N      |  38.8  |       [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r50_fpn_gn_1x_coco.py)       |                   [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/sabl_retinanet_r50_fpn_gn_1x_coco-e16dfcf1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r50_fpn_gn_1x_coco/20200831_141955.log.json)                   |
| SABL RetinaNet | R-101-FPN |   N   |   1x    |      N      |  39.7  |        [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_1x_coco.py)        |                      [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/sabl_retinanet_r101_fpn_1x_coco-42026904.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_1x_coco/20200831_034256.log.json)                      |
| SABL RetinaNet | R-101-FPN |   Y   |   1x    |      N      |  40.5  |      [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_gn_1x_coco.py)       |                 [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/sabl_retinanet_r101_fpn_gn_1x_coco-40a893e8.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_1x_coco/20200830_201422.log.json)                  |
| SABL RetinaNet | R-101-FPN |   Y   |   2x    | Y (640~800) |  42.9  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco-1e63382c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_640_800_coco/20200830_144807.log.json) |
| SABL RetinaNet | R-101-FPN |   Y   |   2x    | Y (480~960) |  43.6  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco-5342f857.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/sabl/sabl_retinanet_r101_fpn_gn_2x_ms_480_960_coco/20200830_164537.log.json) |
# Cascade R-CNN: High Quality Object Detection and Instance Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{Cai_2019,
   title={Cascade R-CNN: High Quality Object Detection and Instance Segmentation},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/tpami.2019.2956516},
   DOI={10.1109/tpami.2019.2956516},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cai, Zhaowei and Vasconcelos, Nuno},
   year={2019},
   pages={1–1}
}
```

## Results and models

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |:------:|:--------:|
|    R-50-FPN     |  caffe  |   1x    |   4.2    |                |  40.4  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco/cascade_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.404_20200504_174853-b857be87.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco/cascade_rcnn_r50_caffe_fpn_1x_coco_20200504_174853.log.json) |
|    R-50-FPN     | pytorch |   1x    |   4.4    |      16.1      |  40.3  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316_214748.log.json) |
|    R-50-FPN     | pytorch |   20e   |  -       |      -         | 41.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco/cascade_rcnn_r50_fpn_20e_coco_20200504_175131.log.json) |
|    R-101-FPN    |  caffe  |   1x    |  6.2     |                | 42.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco/cascade_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.423_20200504_175649-cab8dbd5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_caffe_fpn_1x_coco/cascade_rcnn_r101_caffe_fpn_1x_coco_20200504_175649.log.json) |
|    R-101-FPN    | pytorch |   1x    |   6.4    |      13.5      |  42.0  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317_101744.log.json) |
|    R-101-FPN    | pytorch |   20e   |   -      |      -         |  42.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco/cascade_rcnn_r101_fpn_20e_coco_bbox_mAP-0.425_20200504_231812-5057dcc5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_20e_coco/cascade_rcnn_r101_fpn_20e_coco_20200504_231812.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    |   7.6    |      10.9      |  43.7  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316-95c2deb6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco/cascade_rcnn_x101_32x4d_fpn_1x_coco_20200316_055608.log.json) |
| X-101-32x4d-FPN | pytorch |   20e   |  7.6     |                | 43.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco/cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608-9ae0a720.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_20e_coco/cascade_rcnn_x101_32x4d_fpn_20e_coco_20200906_134608.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |  10.7    |                | 44.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702.log.json) |
| X-101-64x4d-FPN | pytorch |   20e   |  10.7    |                | 44.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357.log.json)|

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :------: | :--------: |
|    R-50-FPN     |  caffe  |   1x    |  5.9     |                | 41.2   | 36.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco/cascade_mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.412__segm_mAP-0.36_20200504_174659-5004b251.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco/cascade_mask_rcnn_r50_caffe_fpn_1x_coco_20200504_174659.log.json) |
|    R-50-FPN     | pytorch |   1x    |  6.0     |  11.2          | 41.2   | 35.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203_170449.log.json) |
|    R-50-FPN     | pytorch |   20e   |  -       | -              | 41.9   | 36.5    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco/cascade_mask_rcnn_r50_fpn_20e_coco_20200504_174711.log.json)|
|    R-101-FPN    |  caffe  |   1x    |  7.8     |                | 43.2   | 37.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_caffe_fpn_1x_coco/cascade_mask_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.432__segm_mAP-0.376_20200504_174813-5c1e9599.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_caffe_fpn_1x_coco/cascade_mask_rcnn_r101_caffe_fpn_1x_coco_20200504_174813.log.json)|
|    R-101-FPN    | pytorch |   1x    |  7.9     |  9.8           | 42.9   | 37.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco/cascade_mask_rcnn_r101_fpn_1x_coco_20200203-befdf6ee.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_1x_coco/cascade_mask_rcnn_r101_fpn_1x_coco_20200203_092521.log.json) |
|    R-101-FPN    | pytorch |   20e   |  -       |  -             | 43.4   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco/cascade_mask_rcnn_r101_fpn_20e_coco_bbox_mAP-0.434__segm_mAP-0.378_20200504_174836-005947da.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r101_fpn_20e_coco/cascade_mask_rcnn_r101_fpn_20e_coco_20200504_174836.log.json)|
| X-101-32x4d-FPN | pytorch |   1x    |  9.2     |  8.6           | 44.3   | 38.3    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201-0f411b1f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco/cascade_mask_rcnn_x101_32x4d_fpn_1x_coco_20200201_052416.log.json) |
| X-101-32x4d-FPN | pytorch |   20e   |  9.2     |   -            | 45.0   | 39.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917-ed1f4751.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco/cascade_mask_rcnn_x101_32x4d_fpn_20e_coco_20200528_083917.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    |  12.2    |  6.7           | 45.3   | 39.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203_044059.log.json) |
| X-101-64x4d-FPN | pytorch |   20e   |  12.2   |                 | 45.6     |39.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033.log.json)|

**Notes:**

- The `20e` schedule in Cascade (Mask) R-CNN indicates decreasing the lr at 16 and 19 epochs, with a total of 20 epochs.
# FoveaBox: Beyond Anchor-based Object Detector

<!-- [ALGORITHM] -->

FoveaBox is an accurate, flexible and completely anchor-free object detection system for object detection framework, as presented in our paper [https://arxiv.org/abs/1904.03797](https://arxiv.org/abs/1904.03797):
Different from previous anchor-based methods, FoveaBox directly learns the object existing possibility and the bounding box coordinates without anchor reference. This is achieved by: (a) predicting category-sensitive semantic maps for the object existing possibility, and (b) producing category-agnostic bounding box for each position that potentially contains an object.

## Main Results

### Results on R50/101-FPN

| Backbone  | Style   |  align  | ms-train| Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | pytorch | N       | N       | 1x      | 5.6      | 24.1           | 36.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219_223025.log.json) |
| R-50      | pytorch | N       | N       | 2x      | 5.6      | -              | 37.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r50_fpn_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203-2df792b1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_2x_coco/fovea_r50_fpn_4x4_2x_coco_20200203_112043.log.json) |
| R-50      | pytorch | Y       | N       | 2x      | 8.1      | 19.4           | 37.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203-8987880d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco/fovea_align_r50_fpn_gn-head_4x4_2x_coco_20200203_134252.log.json) |
| R-50      | pytorch | Y       | Y       | 2x      | 8.1      | 18.3           | 40.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205-85ce26cb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r50_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200205_112557.log.json) |
| R-101     | pytorch | N       | N       | 1x      | 9.2      | 17.4           | 38.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r101_fpn_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219-05e38f1c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_1x_coco/fovea_r101_fpn_4x4_1x_coco_20200219_011740.log.json) |
| R-101     | pytorch | N       | N       | 2x      | 11.7     | -              | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_r101_fpn_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208-02320ea4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r101_fpn_4x4_2x_coco/fovea_r101_fpn_4x4_2x_coco_20200208_202059.log.json) |
| R-101     | pytorch | Y       | N       | 2x      | 11.7     | 14.7           | 40.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208-c39a027a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_4x4_2x_coco/fovea_align_r101_fpn_gn-head_4x4_2x_coco_20200208_203337.log.json) |
| R-101     | pytorch | Y       | Y       | 2x      | 11.7     | 14.7           | 42.0   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208-649c5eb6.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fovea_align_r101_fpn_gn-head_mstrain_640-800_4x4_2x_coco_20200208_202124.log.json) |

[1] *1x and 2x mean the model is trained for 12 and 24 epochs, respectively.* \
[2] *Align means utilizing deformable convolution to align the cls branch.* \
[3] *All results are obtained with a single model and without any test time data augmentation.*\
[4] *We use 4 GPUs for training.*

Any pull requests or issues are welcome.

## Citations

Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.

```latex
@article{kong2019foveabox,
  title={FoveaBox: Beyond Anchor-based Object Detector},
  author={Kong, Tao and Sun, Fuchun and Liu, Huaping and Jiang, Yuning and Shi, Jianbo},
  journal={arXiv preprint arXiv:1904.03797},
  year={2019}
}
```
# FCOS: Fully Convolutional One-Stage Object Detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```

## Results and Models

| Backbone  | Style   | GN      | MS train | Tricks  | DCN     | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | caffe   | Y       | N        | N       | N       | 1x      | 3.6      | 22.7           | 36.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/20201227_180009.log.json) |
| R-50      | caffe   | Y       | N        | Y       | N       | 1x      | 3.7      | -              | 38.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco-0a0d75a8.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco/20210105_135818.log.json)|
| R-50      | caffe   | Y       | N        | Y       | Y       | 1x      | 3.8      | -              | 42.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco-ae4d8b3d.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco/20210105_224556.log.json)|
| R-101     | caffe   | Y       | N        | N       | N       | 1x      | 5.5      | 17.3           | 39.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco/fcos_r101_caffe_fpn_gn-head_1x_coco-0e37b982.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_1x_coco/20210103_155046.log.json) |

| Backbone  | Style   | GN      | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50      | caffe   | Y       | Y        | 2x      | 2.6      | 22.9           | 38.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/20201227_161900.log.json) |
| R-101     | caffe   | Y       | Y        | 2x      | 5.5      | 17.3           | 40.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco/20210103_155046.log.json) |
| X-101     | pytorch | Y       | Y        | 2x      | 10.0     | 9.7            | 42.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco.py) | [model](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth) &#124; [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/20210114_133041.log.json) |

**Notes:**

- The X-101 backbone is X-101-64x4d.
- Tricks means setting `norm_on_bbox`, `centerness_on_reg`, `center_sampling` as `True`.
- DCN means using `DCNv2` in both backbone and head.
# Libra R-CNN: Towards Balanced Learning for Object Detection

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the results in the CVPR 2019 paper [Libra R-CNN](https://arxiv.org/pdf/1904.02701.pdf).

```
@inproceedings{pang2019libra,
  title={Libra R-CNN: Towards Balanced Learning for Object Detection},
  author={Pang, Jiangmiao and Chen, Kai and Shi, Jianping and Feng, Huajun and Ouyang, Wanli and Dahua Lin},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

## Results and models

The results on COCO 2017val are shown in the below table. (results on test-dev are usually slightly higher than val)

| Architecture | Backbone        | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------------:|:---------------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Faster R-CNN | R-50-FPN        | pytorch | 1x      | 4.6      | 19.0           | 38.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
| Fast R-CNN   | R-50-FPN        | pytorch | 1x      |          |                |        | |
| Faster R-CNN | R-101-FPN       | pytorch | 1x      | 6.5      | 14.4           | 40.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203-8dba6a5a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_coco/libra_faster_rcnn_r101_fpn_1x_coco_20200203_001405.log.json) |
| Faster R-CNN | X-101-64x4d-FPN | pytorch | 1x      | 10.8     | 8.5            | 42.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315-3a7d0488.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_coco/libra_faster_rcnn_x101_64x4d_fpn_1x_coco_20200315_231625.log.json) |
| RetinaNet    | R-50-FPN        | pytorch | 1x      | 4.2      | 17.7           | 37.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/libra_rcnn/libra_retinanet_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_retinanet_r50_fpn_1x_coco/libra_retinanet_r50_fpn_1x_coco_20200205_112757.log.json) |
# InstaBoost for MMDetection

<!-- [ALGORITHM] -->

Configs in this directory is the implementation for ICCV2019 paper "InstaBoost: Boosting Instance Segmentation Via Probability Map Guided Copy-Pasting" and provided by the authors of the paper. InstaBoost is a data augmentation method for object detection and instance segmentation. The paper has been released on [`arXiv`](https://arxiv.org/abs/1908.07801).

```latex
@inproceedings{fang2019instaboost,
  title={Instaboost: Boosting instance segmentation via probability map guided copy-pasting},
  author={Fang, Hao-Shu and Sun, Jianhua and Wang, Runzhong and Gou, Minghao and Li, Yong-Lu and Lu, Cewu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={682--691},
  year={2019}
}
```

## Usage

### Requirements

You need to install `instaboostfast` before using it.

```shell
pip install instaboostfast
```

The code and more details can be found [here](https://github.com/GothicAi/Instaboost).

### Integration with MMDetection

InstaBoost have been already integrated in the data pipeline, thus all you need is to add or change **InstaBoost** configurations after **LoadImageFromFile**. We have provided examples like [this](mask_rcnn_r50_fpn_instaboost_4x#L121). You can refer to [`InstaBoostConfig`](https://github.com/GothicAi/InstaBoost-pypi#instaboostconfig) for more details.

## Results and Models

- All models were trained on `coco_2017_train` and tested on `coco_2017_val` for conveinience of evaluation and comparison. In the paper, the results are obtained from `test-dev`.
- To balance accuracy and training time when using InstaBoost, models released in this page are all trained for 48 Epochs. Other training and testing configs strictly follow the original framework.
- For results and models in MMDetection V1.x, please refer to [Instaboost](https://github.com/GothicAi/Instaboost).

|     Network     |       Backbone       | Lr schd | Mem (GB) | Inf time (fps) | box AP  | mask AP | Config |     Download       |
| :-------------: |      :--------:      | :-----: | :------: | :------------: | :------:| :-----: | :------: | :-----------------: |
|    Mask R-CNN   |       R-50-FPN       |   4x    | 4.4      | 17.5           | 40.6    | 36.6    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-d025f83a.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r50_fpn_instaboost_4x_coco/mask_rcnn_r50_fpn_instaboost_4x_coco_20200307_223635.log.json) |
|    Mask R-CNN   |      R-101-FPN       |   4x    | 6.4       |                | 42.5    | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738-f23f3a5f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_r101_fpn_instaboost_4x_coco/mask_rcnn_r101_fpn_instaboost_4x_coco_20200703_235738.log.json) |
|    Mask R-CNN   |   X-101-64x4d-FPN    |   4x    | 10.7     |                | 44.7    | 39.7    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947-8ed58c1b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/instaboost/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco/mask_rcnn_x101_64x4d_fpn_instaboost_4x_coco_20200515_080947.log.json) |
|  Cascade R-CNN  |       R-101-FPN      |   4x    | 6.0      | 12.0            | 43.7    | 38.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307-c19d98d9.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco_20200307_223646.log.json) |
# High-resolution networks (HRNets) for object detection

## Introduction

<!-- [ALGORITHM] -->

```latex
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{SunZJCXLMWLW19,
  title={High-Resolution Representations for Labeling Pixels and Regions},
  author={Ke Sun and Yang Zhao and Borui Jiang and Tianheng Cheng and Bin Xiao
  and Dong Liu and Yadong Mu and Xinggang Wang and Wenyu Liu and Jingdong Wang},
  journal   = {CoRR},
  volume    = {abs/1904.04514},
  year={2019}
}
```

## Results and Models

### Faster R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:| :--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 6.6      | 13.4           | 36.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130_211246.log.json) |
|   HRNetV2p-W18  | pytorch |   2x    | 6.6      |                | 38.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731-a4ec0611.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_2x_coco/faster_rcnn_hrnetv2p_w18_2x_coco_20200702_085731.log.json) |
|   HRNetV2p-W32  | pytorch |   1x    | 9.0      | 12.4           | 40.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130-6e286425.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_1x_coco/faster_rcnn_hrnetv2p_w32_1x_coco_20200130_204442.log.json) |
|   HRNetV2p-W32  | pytorch |   2x    | 9.0        |              | 41.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927-976a9c15.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w32_2x_coco/faster_rcnn_hrnetv2p_w32_2x_coco_20200529_015927.log.json)  |
|   HRNetV2p-W40  | pytorch |   1x    | 10.4     | 10.5           | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210-95c1f5ce.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_1x_coco/faster_rcnn_hrnetv2p_w40_1x_coco_20200210_125315.log.json) |
|   HRNetV2p-W40  | pytorch |   2x    | 10.4     |                |  42.1  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033-0f236ef4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w40_2x_coco/faster_rcnn_hrnetv2p_w40_2x_coco_20200512_161033.log.json)  |

### Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   1x    | 7.0      | 11.7           | 37.7   | 34.2    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205-1c3d78ed.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_1x_coco/mask_rcnn_hrnetv2p_w18_1x_coco_20200205_232523.log.json) |
|   HRNetV2p-W18  | pytorch |   2x    | 7.0      | -              | 39.8   | 36.0    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212-b3c825b1.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w18_2x_coco/mask_rcnn_hrnetv2p_w18_2x_coco_20200212_134222.log.json) |
|   HRNetV2p-W32  | pytorch |   1x    | 9.4      | 11.3           | 41.2   | 37.1    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207-b29f616e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_1x_coco/mask_rcnn_hrnetv2p_w32_1x_coco_20200207_055017.log.json) |
|   HRNetV2p-W32  | pytorch |   2x    | 9.4      | -              | 42.5   | 37.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213-45b75b4d.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco/mask_rcnn_hrnetv2p_w32_2x_coco_20200213_150518.log.json) |
|   HRNetV2p-W40  | pytorch |   1x    |  10.9    |                | 42.1   |  37.5   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646-66738b35.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_1x_coco/mask_rcnn_hrnetv2p_w40_1x_coco_20200511_015646.log.json)  |
|   HRNetV2p-W40  | pytorch |   2x    |   10.9   |                | 42.8   |  38.2   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732-aed5e4ab.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/mask_rcnn_hrnetv2p_w40_2x_coco/mask_rcnn_hrnetv2p_w40_2x_coco_20200512_163732.log.json)  |

### Cascade R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------: | :--------: |
|   HRNetV2p-W18  | pytorch |   20e   |  7.0     | 11.0           | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210_105632.log.json)  |
|   HRNetV2p-W32  | pytorch |   20e   |  9.4     | 11.0           | 43.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208-928455a4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w32_20e_coco/cascade_rcnn_hrnetv2p_w32_20e_coco_20200208_160511.log.json)  |
|   HRNetV2p-W40  | pytorch |   20e   |  10.8    |                | 43.8   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112-75e47b04.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w40_20e_coco/cascade_rcnn_hrnetv2p_w40_20e_coco_20200512_161112.log.json)  |

### Cascade Mask R-CNN

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 8.5      | 8.5            |41.6    |36.4     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210-b543cd2b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco/cascade_mask_rcnn_hrnetv2p_w18_20e_coco_20200210_093149.log.json)  |
|   HRNetV2p-W32  | pytorch |   20e   |          | 8.3            |44.3    |38.6     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043-39d9cf7b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco/cascade_mask_rcnn_hrnetv2p_w32_20e_coco_20200512_154043.log.json)  |
|   HRNetV2p-W40  | pytorch |   20e   | 12.5     |                |45.1    |39.3     |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922-969c4610.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_mask_rcnn_hrnetv2p_w40_20e_coco/cascade_mask_rcnn_hrnetv2p_w40_20e_coco_20200527_204922.log.json)    |

### Hybrid Task Cascade (HTC)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :-------------:|:------:| :------:|:------:|:--------:|
|   HRNetV2p-W18  | pytorch |   20e   | 10.8     | 4.7            | 42.8   | 37.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/htc_hrnetv2p_w18_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210-b266988c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w18_20e_coco/htc_hrnetv2p_w18_20e_coco_20200210_182735.log.json) |
|   HRNetV2p-W32  | pytorch |   20e   | 13.1     | 4.9            | 45.4   | 39.9    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/htc_hrnetv2p_w32_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207-7639fa12.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w32_20e_coco/htc_hrnetv2p_w32_20e_coco_20200207_193153.log.json) |
|   HRNetV2p-W40  | pytorch |   20e   | 14.6     |                | 46.4   | 40.8    | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/htc_hrnetv2p_w40_20e_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411-417c4d5b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/htc_hrnetv2p_w40_20e_coco/htc_hrnetv2p_w40_20e_coco_20200529_183411.log.json) |

### FCOS

| Backbone  | Style   |  GN     | MS train | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:-------:|:------:|:------:|:------:|:------:|:--------:|
|HRNetV2p-W18| pytorch | Y       | N       | 1x       | 13.0 | 12.9 | 35.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710-4ad151de.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco/fcos_hrnetv2p_w18_gn-head_4x4_1x_coco_20201212_100710.log.json) |
|HRNetV2p-W18| pytorch | Y       | N       | 2x       | 13.0 | -    | 38.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110-5c575fa5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_4x4_2x_coco_20201212_101110.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 1x       | 17.5 | 12.9 | 39.5   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730-cb8055c0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco/fcos_hrnetv2p_w32_gn-head_4x4_1x_coco_20201211_134730.log.json) |
|HRNetV2p-W32| pytorch | Y       | N       | 2x       | 17.5 | -    | 40.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133-77b6b9bb.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_4x4_2x_coco_20201212_112133.log.json) |
|HRNetV2p-W18| pytorch | Y       | Y       | 2x       | 13.0 | 12.9 | 38.3   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651-441e9d9f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w18_gn-head_mstrain_640-800_4x4_2x_coco_20201212_111651.log.json) |
|HRNetV2p-W32| pytorch | Y       | Y       | 2x       | 17.5 | 12.4 | 41.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846-b6f2b49f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w32_gn-head_mstrain_640-800_4x4_2x_coco_20201212_090846.log.json) |
|HRNetV2p-W48| pytorch | Y       | Y       | 2x       | 20.3 | 10.8 | 42.7   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752-f22d2ce5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/hrnet/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco/fcos_hrnetv2p_w40_gn-head_mstrain_640-800_4x4_2x_coco_20201212_124752.log.json) |

**Note:**

- The `28e` schedule in HTC indicates decreasing the lr at 24 and 27 epochs, with a total of 28 epochs.
- HRNetV2 ImageNet pretrained models are in [HRNets for Image Classification](https://github.com/HRNet/HRNet-Image-Classification).
# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## Introduction

<!-- [ALGORITHM] -->

```latex
@article{Ren_2017,
   title={Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
   year={2017},
   month={Jun},
}
```

## Results and models

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-DC5     |  caffe  |   1x    | -        | -              | 37.2   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909-531f0f43.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_1x_coco/faster_rcnn_r50_caffe_dc5_1x_coco_20201030_151909.log.json) |
|    R-50-FPN     |  caffe  |   1x    | 3.8      |                | 37.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.378_20200504_180032-c5925ee5.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_coco/faster_rcnn_r50_caffe_fpn_1x_coco_20200504_180032.log.json) |
|    R-50-FPN     | pytorch |   1x    | 4.0      | 21.4           | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|    R-50-FPN     | pytorch |   2x    | -        | -              | 38.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_20200504_210434.log.json) |
|    R-101-FPN    |  caffe  |   1x    | 5.7      |                | 39.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_20200504_180057.log.json) |
|    R-101-FPN    | pytorch |   1x    | 6.0      | 15.6           | 39.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130_204655.log.json) |
|    R-101-FPN    | pytorch |   2x    | -        | -              | 39.8   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_20200504_210455.log.json) |
| X-101-32x4d-FPN | pytorch |   1x    | 7.2      | 13.8           | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203_000520.log.json) |
| X-101-32x4d-FPN | pytorch |   2x    | -        | -              | 41.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_bbox_mAP-0.412_20200506_041400-64a12c0b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco/faster_rcnn_x101_32x4d_fpn_2x_coco_20200506_041400.log.json) |
| X-101-64x4d-FPN | pytorch |   1x    | 10.3     | 9.4            | 42.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204_134340.log.json) |
| X-101-64x4d-FPN | pytorch |   2x    | -        | -              | 41.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033.log.json)  |

## Different regression loss

We trained with R-50-FPN pytorch style backbone for 1x schedule.

|    Backbone     | Loss type | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-------: | :------: | :------------: | :----: | :------: | :--------: |
|    R-50-FPN     |  L1Loss   | 4.0      | 21.4           | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130_204655.log.json) |
|    R-50-FPN     |  IoULoss  |          |                | 37.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_iou_1x_coco-fdd207f3.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_iou_1x_coco_20200506_095954.log.json)  |
|    R-50-FPN     |  GIoULoss |          |                | 37.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_giou_1x_coco-0eada910.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_giou_1x_coco_20200505_161120.log.json)  |
|    R-50-FPN     |  BoundedIoULoss |          |                | 37.4   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_bounded_iou_1x_coco-98ad993b.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_bounded_iou_1x_coco_20200505_160738.log.json)  |

## Pre-trained Models

We also train some models with longer schedules and multi-scale training. The users could finetune them for downstream tasks.

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :------: | :--------: |
|    [R-50-DC5](./faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py)          |  caffe  |   1x    | -        |                | 37.4   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco_20201028_233851-b33d21b9.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco/faster_rcnn_r50_caffe_dc5_mstrain_1x_coco_20201028_233851.log.json)
|    [R-50-DC5](./faster_rcnn_r50_caffe_dc5_mstrain_3x_coco.py)          |  caffe  |   3x    | -        |                | 38.7   | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_3x_coco.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_3x_coco/faster_rcnn_r50_caffe_dc5_mstrain_3x_coco_20201028_002107-34a53b2c.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_dc5_mstrain_3x_coco/faster_rcnn_r50_caffe_dc5_mstrain_3x_coco_20201028_002107.log.json)
|    [R-50-FPN](./faster_rcnn_r50_caffe_fpn_mstrain_2x_coco.py)     |  caffe  |   2x    | 4.3      |                | 39.7   |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco_bbox_mAP-0.397_20200504_231813-10b2de58.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco/faster_rcnn_r50_caffe_fpn_mstrain_2x_coco_20200504_231813.log.json)
|    [R-50-FPN](./faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py)     |  caffe  |   3x    | 4.3      |                | 40.2   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20200504_163323.log.json)

We further finetune some pre-trained models on the COCO subsets, which only contain only a few of the 80 categories.

| Backbone                                                     | Style | Class name         | Pre-traind model                                             | Mem (GB) | box AP | Config                                                       | Download                                                     |
| ------------------------------------------------------------ | ----- | ------------------ | ------------------------------------------------------------ | -------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [R-50-FPN](./faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py)          | caffe | person             | [R-50-FPN-Caffe-3x](./faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py) | 3.7      | 55.8   | [config](./faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929.log.json)                                                 |
| [R-50-FPN](./faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person-bicycle-car.py) | caffe | person-bicycle-car | [R-50-FPN-Caffe-3x](./faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py) | 3.7      | 44.1   | [config](./faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person-bicycle-car.py) | [model](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117-6eda6d92.pth) &#124; [log](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car/faster_rcnn_r50_fpn_1x_coco-person-bicycle-car_20201216_173117.log.json) |
# Mixed Precision Training

## Introduction

<!-- [OTHERS] -->

```latex
@article{micikevicius2017mixed,
  title={Mixed precision training},
  author={Micikevicius, Paulius and Narang, Sharan and Alben, Jonah and Diamos, Gregory and Elsen, Erich and Garcia, David and Ginsburg, Boris and Houston, Michael and Kuchaiev, Oleksii and Venkatesh, Ganesh and others},
  journal={arXiv preprint arXiv:1710.03740},
  year={2017}
}
```

## Results and Models

| Architecture | Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN | R-50      | pytorch | 1x      | 3.4      | 28.8           | 37.5   | -       |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/faster_rcnn_r50_fpn_fp16_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fp16/faster_rcnn_r50_fpn_fp16_1x_coco/faster_rcnn_r50_fpn_fp16_1x_coco_20200204_143530.log.json) |
| Mask   R-CNN | R-50      | pytorch | 1x      | 3.6      | 24.1           | 38.1   | 34.7    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/mask_rcnn_r50_fpn_fp16_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_1x_coco/mask_rcnn_r50_fpn_fp16_1x_coco_20200205-59faf7e4.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_1x_coco/mask_rcnn_r50_fpn_fp16_1x_coco_20200205_130539.log.json) |
| Retinanet    | R-50      | pytorch | 1x      | 2.8      | 31.6           | 36.4  |     |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fp16/retinanet_r50_fpn_fp16_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702_020127.log.json) |
# Albu Example

<!-- [OTHERS] -->

```
@article{2018arXiv180906839B,
  author = {A. Buslaev, A. Parinov, E. Khvedchenya, V.~I. Iglovikov and A.~A. Kalinin},
  title = "{Albumentations: fast and flexible image augmentations}",
  journal = {ArXiv e-prints},
  eprint = {1809.06839},
  year = 2018
}
```

## Results and Models

| Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP | Config | Download |
|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:-------:|:------:|:--------:|
| R-50      | pytorch | 1x      | 4.4      | 16.6           |  38.0  | 34.5    |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/albu_example/mask_rcnn_r50_fpn_albu_1x_coco/mask_rcnn_r50_fpn_albu_1x_coco_20200208-ab203bcd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/albu_example/mask_rcnn_r50_fpn_albu_1x_coco/mask_rcnn_r50_fpn_albu_1x_coco_20200208_225520.log.json) |
# DeepFashion

<!-- [DATASET] -->

[MMFashion](https://github.com/open-mmlab/mmfashion) develops "fashion parsing and segmentation" module
based on the dataset
[DeepFashion-Inshop](https://drive.google.com/drive/folders/0B7EVK8r0v71pVDZFQXRsMDZCX1E?usp=sharing).
Its annotation follows COCO style.
To use it, you need to first download the data. Note that we only use "img_highres" in this task.
The file tree should be like this:

```sh
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── DeepFashion
│   │   ├── In-shop
│   │   ├── Anno
│   │   │   ├── segmentation
│   │   │   |   ├── DeepFashion_segmentation_train.json
│   │   │   |   ├── DeepFashion_segmentation_query.json
│   │   │   |   ├── DeepFashion_segmentation_gallery.json
│   │   │   ├── list_bbox_inshop.txt
│   │   │   ├── list_description_inshop.json
│   │   │   ├── list_item_inshop.txt
│   │   │   └── list_landmarks_inshop.txt
│   │   ├── Eval
│   │   │   └── list_eval_partition.txt
│   │   ├── Img
│   │   │   ├── img
│   │   │   │   ├──XXX.jpg
│   │   │   ├── img_highres
│   │   │   └── ├──XXX.jpg

```

After that you can train the Mask RCNN r50 on DeepFashion-In-shop dataset by launching training with the `mask_rcnn_r50_fpn_1x.py` config
or creating your own config file.

```
@inproceedings{liuLQWTcvpr16DeepFashion,
   author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
   title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
   booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   month = {June},
   year = {2016}
}
```

## Model Zoo

|   Backbone  |  Model type  |       Dataset       |  bbox detection Average Precision  | segmentation Average Precision |  Config |      Download (Google)      |
| :---------: | :----------: | :-----------------: | :--------------------------------: | :----------------------------: | :---------:| :-------------------------: |
|   ResNet50  |   Mask RCNN  | DeepFashion-In-shop |                0.599               |              0.584             |[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/deepfashion/mask_rcnn_r50_fpn_15e_deepfashion.py)|  [model](https://drive.google.com/open?id=1q6zF7J6Gb-FFgM87oIORIt6uBozaXp5r) &#124; [log](https://drive.google.com/file/d/1qTK4Dr4FFLa9fkdI6UVko408gkrfTRLP/view?usp=sharing)   |
# Feature Selective Anchor-Free Module for Single-Shot Object Detection

<!-- [ALGORITHM] -->

FSAF is an anchor-free method published in CVPR2019 ([https://arxiv.org/pdf/1903.00621.pdf](https://arxiv.org/pdf/1903.00621.pdf)).
Actually it is equivalent to the anchor-based method with only one anchor at each feature map position in each FPN level.
And this is how we implemented it.
Only the anchor-free branch is released for its better compatibility with the current framework and less computational budget.

In the original paper, feature maps within the central 0.2-0.5 area of a gt box are tagged as ignored. However,
it is empirically found that a hard threshold (0.2-0.2) gives a further gain on the performance. (see the table below)

## Main Results

### Results on R50/R101/X101-FPN

| Backbone   |  ignore range | ms-train| Lr schd |Train Mem (GB)| Train time (s/iter) | Inf time (fps) | box AP | Config | Download |
|:----------:|  :-------:    |:-------:|:-------:|:------------:|:---------------:|:--------------:|:-------------:|:------:|:--------:|
| R-50       |   0.2-0.5     | N       | 1x      |    3.15      | 0.43            |    12.3        | 36.0 (35.9)   |  | [model](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200715-b555b0e0.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco/fsaf_pscale0.2_nscale0.5_r50_fpn_1x_coco_20200715_094657.log.json)  |
| R-50       |   0.2-0.2     | N       | 1x      |    3.15      | 0.43            |    13.0        | 37.4          | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_r50_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco_20200428_072327.log.json)|
| R-101      |   0.2-0.2     | N       | 1x      |    5.08      | 0.58            |    10.8        | 39.3 (37.9)   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_r101_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco-9e71098f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r101_fpn_1x_coco/fsaf_r101_fpn_1x_coco_20200428_160348.log.json)|
| X-101      |   0.2-0.2     | N       | 1x      |    9.38      | 1.23            |    5.6         | 42.4 (41.0)   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fsaf/fsaf_x101_64x4d_fpn_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco-e3f6e6fd.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_x101_64x4d_fpn_1x_coco/fsaf_x101_64x4d_fpn_1x_coco_20200428_160424.log.json)|

**Notes:**

- *1x means the model is trained for 12 epochs.*
- *AP values in the brackets represent those reported in the original paper.*
- *All results are obtained with a single model and single-scale test.*
- *X-101 backbone represents ResNext-101-64x4d.*
- *All pretrained backbones use pytorch style.*
- *All models are trained on 8 Titan-XP gpus and tested on a single gpu.*

## Citations

BibTeX reference is as follows.

```latex
@inproceedings{zhu2019feature,
  title={Feature Selective Anchor-Free Module for Single-Shot Object Detection},
  author={Zhu, Chenchen and He, Yihui and Savvides, Marios},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={840--849},
  year={2019}
}
```
# PASCAL VOC Dataset

<!-- [DATASET] -->

```
@Article{Everingham10,
   author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
   title = "The Pascal Visual Object Classes (VOC) Challenge",
   journal = "International Journal of Computer Vision",
   volume = "88",
   year = "2010",
   number = "2",
   month = jun,
   pages = "303--338",
}
```

## Results and Models

| Architecture | Backbone  | Style   | Lr schd | Mem (GB) | Inf time (fps) | box AP | Config | Download |
|:------------:|:---------:|:-------:|:-------:|:--------:|:--------------:|:------:|:------:|:--------:|
| Faster R-CNN | R-50      | pytorch | 1x      | 2.6   | -          | 79.5  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712/20200623_015208.log.json) |
| Retinanet    | R-50      | pytorch | 1x      | 2.1   | -          | 77.3  |[config](https://github.com/open-mmlab/mmdetection/tree/master/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/pascal_voc/retinanet_r50_fpn_1x_voc0712/retinanet_r50_fpn_1x_voc0712_20200616_014642.log.json) |
# CARAFE: Content-Aware ReAssembly of FEatures

## Introduction

<!-- [ALGORITHM] -->

We provide config files to reproduce the object detection & instance segmentation results in the ICCV 2019 Oral paper for [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188).

```
@inproceedings{Wang_2019_ICCV,
    title = {CARAFE: Content-Aware ReAssembly of FEatures},
    author = {Wang, Jiaqi and Chen, Kai and Xu, Rui and Liu, Ziwei and Loy, Chen Change and Lin, Dahua},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Results and Models

The results on COCO 2017 val is shown in the below table.

| Method               | Backbone | Style   | Lr schd | Test Proposal Num | Inf time (fps) | Box AP | Mask AP | Config | Download |
|:--------------------:|:--------:|:-------:|:-------:|:-----------------:|:--------------:|:------:|:-------:|:------:|:--------:|
| Faster R-CNN w/ CARAFE | R-50-FPN | pytorch | 1x      | 1000 | 16.5 | 38.6   | 38.6       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/carafe/faster_rcnn_r50_fpn_carafe_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/carafe/faster_rcnn_r50_fpn_carafe_1x_coco/faster_rcnn_r50_fpn_carafe_1x_coco_20200504_175733.log.json) |
| -                      |    -     |  -      | -       | 2000 |      |        |            |  |
| Mask R-CNN w/ CARAFE   | R-50-FPN | pytorch | 1x      | 1000 | 14.0 | 39.3   | 35.8       | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.393__segm_mAP-0.358_20200503_135957-8687f195.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/carafe/mask_rcnn_r50_fpn_carafe_1x_coco/mask_rcnn_r50_fpn_carafe_1x_coco_20200503_135957.log.json) |
| -                      |   -      |  -      |   -     | 2000 |      |        |            |  |

## Implementation

The CUDA implementation of CARAFE can be find at https://github.com/myownskyW7/CARAFE.
