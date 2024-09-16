# SC2: Supervised Compression for Split Computing

## Implemented Methods
1. CR + BQ: ["Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks"](https://arxiv.org/abs/2007.15818)
2. End-to-End: ["End-to-end Learning of Compressible Features"](https://arxiv.org/abs/2007.11797) 
3. Entropic Student: ["Supervised Compression for Resource-Constrained Edge Computing Systems"](https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html)

## Test without training step
If you have checkpoints (e.g., trained model weights available [here](https://github.com/yoshitomo-matsubara/sc2-benchmark#checkpoints)), 
you can skip the training step and just test the (pre)trained models simply by adding `-test_only` to the following commands.

e.g.,
```shell
python legacy/script/task/image_classification.py -test_only -student_only \
  --config legacy/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq12ch_from_resnet50.yaml
```

## 1. ImageNet (ILSVRC 2012): Image Classification
The following examples use ResNet-50 as a reference model. More examples are available in [configs/](https://github.com/yoshitomo-matsubara/sc2-benchmark/tree/main/configs).

### 1.1 CR + BQ
```shell
for bch in 1 2 3 6 9 12; do
  python legacy/script/task/image_classification.py -student_only \
    --config legacy/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${bch}ch_from_resnet50.yaml \
    --log legacy/log/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${bch}ch_from_resnet50.txt
done
```

### 1.2 End-to-End
```shell
for beta in 1.28e-8 1.024e-7 2.048e-7 8.192e-7 3.2768e-6; do 
  python legacy/script/task/image_classification.py \
    --config legacy/configs/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}.yaml \
    --log legacy/log/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}.txt
done
```

### 1.3 Entropic Student
```shell
for beta in 0.08 0.16 0.32 0.64 1.28 2.56 5.12; do 
  python legacy/script/task/image_classification.py -student_only \
    --config legacy/configs/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${beta}_from_resnet50.yaml \
    --log legacy/log/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${beta}_from_resnet50.txt
done
```

---

## 2. COCO 2017: Object Detection
The following examples use Faster R-CNN with ResNet-50 and FPN as a reference model.

### 2.1 CR + BQ

```shell
for bch in 1 2 3 6 9 12; do
  python legacy/script/task/object_detection.py -student_only \
    --config legacy/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_splittable_resnet50-bq${bch}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml \
    --log legacy/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_splittable_resnet50-bq${bch}ch_fpn_from_faster_rcnn_resnet50_fpn.txt
done
```

### 2.2 End-to-End
```shell
for beta in 1.28e-8 1.024e-7 2.048e-7 8.192e-7 3.2768e-6; do 
  python legacy/script/task/object_detection.py \
    --config legacy/configs/coco2017/supervised_compression/end-to-end/faster_rcnn_splittable_resnet50-fp-beta${beta}_fpn.yaml \
    --log legacy/log/coco2017/supervised_compression/end-to-end/faster_rcnn_splittable_resnet50-fp-beta${beta}_fpn.txt
done
```

### 2.3 Entropic Student
```shell
for beta in 0.08 0.16 0.32 0.64 1.28 2.56 5.12; do 
  python legacy/script/task/object_detection.py -student_only \
    --config legacy/configs/coco2017/supervised_compression/entropic_student/faster_rcnn_splittable_resnet50-fp-beta${beta}_fpn_from_faster_rcnn_resnet50_fpn.yaml \
    --log legacy/log/coco2017/supervised_compression/entropic_student/faster_rcnn_splittable_resnet50-fp-beta${beta}_fpn_from_faster_rcnn_resnet50_fpn.txt
done
```

---

## 3. PASCAL VOC 2012: Semantic Segmentation
The following examples use DeepLabv3 with ResNet-50 as a reference model.

### 3.1 CR + BQ

```shell
for bch in 1 2 3 6 9 12; do
  python legacy/script/task/semantic_segmentation.py -student_only \
    --config legacy/configs/pascal_voc2012/supervised_compression/ghnd-bq/deeplabv3_resnet50-bq${bch}ch_from_deeplabv3_resnet50.yaml \
    --log legacy/log/pascal_voc2012/supervised_compression/ghnd-bq/deeplabv3_resnet50-bq${bch}ch_from_deeplabv3_resnet50.txt
done
```

### 3.2 End-to-End
```shell
for beta in 1.28e-8 1.024e-7 2.048e-7 8.192e-7 3.2768e-6; do 
  python legacy/script/task/semantic_segmentation.py \
    --config legacy/configs/pascal_voc2012/supervised_compression/end-to-end/deeplabv3_splittable_resnet50-fp-beta${beta}.yaml \
    --log legacy/log/pascal_voc2012/supervised_compression/end-to-end/deeplabv3_splittable_resnet50-fp-beta${beta}.txt
done
```

### 3.3 Entropic Student
```shell
for beta in 0.16 0.32 0.64 1.28 2.56 5.12; do 
  python legacy/script/task/semantic_segmentation.py -student_only \
    --config legacy/configs/pascal_voc2012/supervised_compression/entropic_student/deeplabv3_splittable_resnet50-fp-beta${beta}_from_deeplabv3_resnet50.yaml \
    --log legacy/log/pascal_voc2012/supervised_compression/entropic_student/deeplabv3_splittable_resnet50-fp-beta${beta}_from_deeplabv3_resnet50.txt
done
```
