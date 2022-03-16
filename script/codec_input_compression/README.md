# Codec-based Input Compression Baselines

We considered the following codec-based image compression methods:
- JPEG
- WebP
- BPG

If you want to use BPG, you will need to manually install the software 
```shell
bash script/software/install_bpg.sh
```

The script will place the encoder and decoder in `~/software/`

## ImageNet (ILSVRC 2012): Image Classification
Codec-based input compression followed by ResNet-50

```shell
bash script/codec_input_compression/ilsvrc2012-image_classification.sh jpeg-resnet50 jpeg
bash script/codec_input_compression/ilsvrc2012-image_classification.sh webp-resnet50 webp
bash script/codec_input_compression/ilsvrc2012-image_classification.sh bpg-resnet50 bpg 5 5 50
```

## COCO 2017: Object Detection
Codec-based input compression followed by Faster R-CNN with ResNet-50 and FPN

```shell
bash script/codec_input_compression/coco2017-object_detection.sh jpeg-faster_rcnn_resnet50_fpn jpeg
bash script/codec_input_compression/coco2017-object_detection.sh webp-faster_rcnn_resnet50_fpn webp
bash script/codec_input_compression/coco2017-object_detection.sh bpg-faster_rcnn_resnet50_fpn bpg 5 5 50
```

## PASCAL VOC 2012: Semantic Segmentation
Codec-based input compression followed by DeepLabv3 with ResNet-50

```shell
bash script/codec_input_compression/pascal_voc2012-semantic_segmentation.sh jpeg-deeplabv3_resnet50 jpeg
bash script/codec_input_compression/pascal_voc2012-semantic_segmentation.sh webp-deeplabv3_resnet50 webp
bash script/codec_input_compression/pascal_voc2012-semantic_segmentation.sh bpg-deeplabv3_resnet50 bpg 5 5 50
```
