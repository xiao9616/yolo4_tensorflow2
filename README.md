# yolo4_tensorflow2
yolo 4th edition  implemented by tensorflow2.0

[TOC]



## 综述

![img](README.assets/640.webp)

对于那些在GPU平台上运行的检测器，它们的主干网络可能为VGG、ResNet、ResNeXt或DenseNet。

而对于那些在CPU平台上运行的检测器，他们的检测器可能为SqueezeNet ，MobileNet， ShufflfleNet。

最具代表性的二阶段目标检测器R-CNN系列，包括fast R-CNN，faster R-CNN ，R-FCN [9]，Libra R-CNN。也可以使得二阶段目标检测器成为anchor-free目标检测器，例如RepPoints。至于一阶段目标检测器，最具代表性的网络包括YOLO、SSD、RetinaNet。

一阶段的anchor-free目标检测器在不断发展，包括CenterNet、CornerNet、FCOS等。在近些年来的发展，目标检测器通常是在头部和主干网络之间加入一些层，这些层被用来收集不同阶段的特征图,拥有这种机制的网络包括Feature Pyramid Network (FPN)，Path Aggregation Network (PAN)，BiFPN和NAS-FPN。

除了上述模型外，一些研究人员将重心放在了研究主干网络上（DetNet，DetNAS），而还有一些研究人员则致力于构建用于目标检测的新的模型（SpineNet，HitDetector）。

总的来说，一般的目标检测器由以下几部分组成：

- Input: Image, Patches, Image Pyramid

- Backbones: VGG16, ResNet-50,SpineNet，EffificientNet-B0/B7, CSPResNeXt50， CSPDarknet53

- Neck:

- Additional blocks: SPP,ASPP,RFB，SAM

- Path-aggregation blocks: FPN，PAN，NAS-FPN，Fully-connected FPN，BiFPN，ASFF，SFAM

- Heads:

- Dense Prediction(one-stage):

  - RPN，SSD，YOLO，RetinaNet(anchor based)

  - CornerNet，CenterNet，MatrixNet，FCOS(anchor free)

-  Sparse Prediction(two-stage):

- - Faster R-CNN，R-FCN，Mask R-CNN(anchor based)
  - RepPoints（anchor free）

### BOF

Bag of freebies：通常，传统的目标检测器都是离线训练的。因此，研究人员经常喜欢利用这一优点来开发一种更好的训练方法，使得在不增加推理成本的情况下提升目标检测器的精度。我们称这些仅仅改变了训练策略或增加了训练成本的方法为“Bag of freebies”。目前通常采用的方法使数据增强。数据增强的目的是增加输入图像的灵活性，因此设计的目标检测器有着更高的鲁棒性。

### BOS

Bag of specials：对于那些增加模块和后处理的方法，它们仅仅少量的增加了推理成本，却能够显著地提升目标检测精度，我们称之为“bag of specials”。一般说来，这些插入的模块是用来增强某些属性，例如增强感受野，引入警示机制或增强特征的集成能力等。而后处理则用来筛选模型的预测结果。

### Selection of BoF and BoS

为了改进目标检测训练，CNN通常使用如下内容：

- Activations: ReLU, leaky-ReLU, parametric-ReLU,ReLU6, SELU, Swish, or Mish
- Bounding box regression loss: MSE, IoU, GIoU,CIoU, DIoU
- Data augmentation: CutOut, MixUp, CutMix
- Regularization method: DropOut, DropPath，Spatial DropOut [79], or DropBlock
- Normalization of the network activations by their mean and variance: Batch Normalization (BN) ，Cross-GPU Batch Normalization (CGBN or SyncBN)，Filter Response Normalization (FRN) [70], or Cross-Iteration Batch Normalization (CBN)
- Skip-connections: Residual connections, Weighted residual connections, Multi-input weighted residual connections, or Cross stage partial connections (CSP)

### Additional improvements

为了使我们设计的目标检测器更加适合在单GPU上训练，我们有如下的设计和改进：

- 引入Mosaic, and Self-Adversarial Training (SAT)的数据增广方式
- 应用遗传算法选择最优超参数
- 修改现有的方法使之更适合训练和检测- modifified SAM, modifified PAN, and Cross mini-Batch Normalization (CmBN)

Mosaic代表了一种新的数据增广方式。它混合了四张训练图像。因此，四种不同的上下文被混合，而CutMix只混合两个输入图像。这使得能够检测正常背景之外的目标。

Self-Adversarial Training (SAT)同样代表了一种数据混合方式，分两个阶段进行。

CmBN为CBN的修改版，定义为Cross mini-Batch Normalization（CmBN），它仅仅在一个batch之中收集小批量统计信息。

我们将SAM从spatial-wise attention改为point-wise attention。同时，替换PAN到concatenation的连接。

## 论文研读

yolo4的主要改进点：

### 1.WRC



### 2.CSP





### 3.CmBN



### 4.SAT



### 5.Mish



### 6.Mosaic data augmnetation



### 7.DropBlock regularization



### 8.CIOU loss



## 网络结构

我们在CSPDarknet53上添加SPP block，因为它能够显著增加感受野，分离出最重要的上下文特征且几乎没有降低网络的运行速度。我们使用PANet作为参数聚集的方法，而不是YOLOv3中使用的FPN。最后，我们使用CSPDarknet53的主干网络, SPP附加模块, PANet 以及YOLOv3的头部作为YOLOv4的框架。

### YOLOv4 consists of:

- Backbone: CSPDarknet53
- Neck: SPP, PAN
- Head: YOLOv3

### YOLO v4 uses:

- Bag of Freebies (BoF) for backbone: CutMix and Mosaic data augmentation, DropBlock regularization,Class label smoothing
- Bag of Specials (BoS) for backbone: Mish activation, Cross-stage partial connections (CSP), Multi-input weighted residual connections (MiWRC)
- Bag of Freebies (BoF) for detector: CIoU-loss,CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training, Eliminate grid sensitivity, Using multiple anchors for a single ground truth, Cosine annealing scheduler, Optimal hyper parameters, Random training shapes
- Bag of Specials (BoS) for detector: Mish activation,SPP-block, SAM-block, PAN path-aggregation block, DIoU-NMS

## 代码实现

