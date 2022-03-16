# Neural Input Compression Baselines

We considered the following neural image compression models:
- Factorized Prior
- Scale Hyperprior
- Mean-scale Hyperprior
- Joint Autoregressive Hierarchical Prior


## ImageNet (ILSVRC 2012): Image Classification
Neural input compression followed by ResNet-50

```shell
bash script/neural_input_compression/ilsvrc2012-image_classification.sh factorized_prior-resnet50 8
bash script/neural_input_compression/ilsvrc2012-image_classification.sh scale_hyperprior-resnet50 8
bash script/neural_input_compression/ilsvrc2012-image_classification.sh mean_scale_hyperprior-resnet50 8
bash script/neural_input_compression/ilsvrc2012-image_classification.sh joint_autoregressive_hierarchical_prior-resnet50 8
```

## COCO 2017: Object Detection
Neural input compression followed by Faster R-CNN with ResNet-50 and FPN

```shell
bash script/neural_input_compression/coco2017-object_detection.sh factorized_prior-faster_rcnn_resnet50_fpn 8
bash script/neural_input_compression/coco2017-object_detection.sh scale_hyperprior-faster_rcnn_resnet50_fpn 8
bash script/neural_input_compression/coco2017-object_detection.sh mean_scale_hyperprior-faster_rcnn_resnet50_fpn 8
bash script/neural_input_compression/coco2017-object_detection.sh joint_autoregressive_hierarchical_prior-faster_rcnn_resnet50_fpn 8
```

## PASCAL VOC 2012: Semantic Segmentation
Neural input compression followed by DeepLabv3 with ResNet-50

```shell
bash script/neural_input_compression/pascal_voc2012-semantic_segmentation.sh factorized_prior-deeplabv3_resnet50 8
bash script/neural_input_compression/pascal_voc2012-semantic_segmentation.sh scale_hyperprior-deeplabv3_resnet50 8
bash script/neural_input_compression/pascal_voc2012-semantic_segmentation.sh mean_scale_hyperprior-deeplabv3_resnet50 8
bash script/neural_input_compression/pascal_voc2012-semantic_segmentation.sh joint_autoregressive_hierarchical_prior-deeplabv3_resnet50 8
```
