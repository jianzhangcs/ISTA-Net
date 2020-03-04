# ISTA-Net
##### ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing, CVPR2018 (Tensorflow Code)

## [[New PyTorch Version]](https://github.com/jianzhangcs/ISTA-Net-PyTorch)

This repository is for ISTA-Net introduced in the following paper:

[Jian Zhang](http://jianzhang.tech/) and [Bernard Ghanem](http://www.bernardghanem.com/), "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing", CVPR 2018, [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ISTA-Net_Interpretable_Optimization-Inspired_CVPR_2018_paper.pdf) 

Training data ([Training_Data_Img91.mat](https://drive.google.com/open?id=1AoEcNA5-onnSqBcWZawNw7ZFrJ1fFR_C)) and other training models can be downloaded at [GoogleDrive](https://drive.google.com/open?id=1AoEcNA5-onnSqBcWZawNw7ZFrJ1fFR_C). 

The code is tested on both Windows and Linux environments (Tensorflow: 1.2.0, CUDA8.0, cuDNN5.1) with Titan 1080Ti GPU.

## Introduction
With the aim of developing a fast yet accurate algorithm for compressive sensing (CS) reconstruction of natural images, we combine in this paper the merits of two existing categories of CS methods: the structure insights of traditional optimization-based methods and the speed of recent network-based ones. Specifically, we propose a novel structured deep network, dubbed ISTA-Net, which is inspired by the Iterative Shrinkage-Thresholding Algorithm (ISTA) for optimizing a general L1 norm CS reconstruction model. To cast ISTA into deep network form, we develop an effective strategy to solve the proximal mapping associated with the sparsity-inducing regularizer using nonlinear transforms. All the parameters in ISTA-Net (\eg nonlinear transforms, shrinkage thresholds, step sizes, etc.) are learned end-to-end, rather than being hand-crafted. Moreover, considering that the residuals of natural images are more compressible, an enhanced version of ISTA-Net in the residual domain, dubbed ISTA-Net+, is derived to further improve CS reconstruction. Extensive CS experiments demonstrate that the proposed ISTA-Nets outperform existing state-of-the-art optimization-based and network-based CS methods by large margins, while maintaining fast computational speed.

![ISTA-Net](/Figs/ista_phase.png)
Figure 1. Illustration of the proposed ISTA-Net framework.

## Citation
If you find our code helpful in your resarch or work, please cite our paper.
```
@inproceedings{zhang2018ista,
  title={ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing},
  author={Zhang, Jian and Ghanem, Bernard},
  booktitle={CVPR},
  pages={1828--1837},
  year={2018}
}
```

