# Datasets

Download and preprocess datasets before you run experiments.  
Here, we provide three examples: ImageNet (ILSVRC 2012), COCO 2017, and PASCAL VOC 2012.

## 1. ImageNet (ILSVRC 2012): Image Classification
### 1.1 Download the datasets
As the terms of use do not allow to distribute the URLs, you will have to create an account [here](http://image-net.org/download) to get the URLs, and replace `${TRAIN_DATASET_URL}` and `${VAL_DATASET_URL}` with them.
```shell
wget ${TRAIN_DATASET_URL} ./
wget ${VAL_DATASET_URL} ./
```

### 1.2 Untar and extract files
```shell
# Go to the root of this repository
mkdir ~/dataset/ilsvrc2012/{train,val} -p
mv ILSVRC2012_img_train.tar ~/dataset/ilsvrc2012/train/
mv ILSVRC2012_img_val.tar ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/train/
tar -xvf ILSVRC2012_img_train.tar
mv ILSVRC2012_img_train.tar ../
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xf ../$f)
done
rm -r *.tar
cd ../../../../

wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/val/
tar -xvf ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar ../
sh valprep.sh
mv valprep.sh ../
cd ../../../../
```


## 2. COCO 2017: Object Detection
### 2.1 Download the datasets
```shell
wget http://images.cocodataset.org/zips/train2017.zip ./
wget http://images.cocodataset.org/zips/val2017.zip ./
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ./
```

### 2.2 Unzip and extract files
```shell
# Go to the root of this repository
mkdir ~/dataset/coco2017/ -p
mv train2017.zip ~/dataset/coco2017/
mv val2017.zip ~/dataset/coco2017/
mv annotations_trainval2017.zip ~/dataset/coco2017/
cd ~/dataset/coco2017/
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ../../../
```


## 3. PASCAL VOC 2012: Semantic Segmentation
You can skip Steps 3.1 and 3.2 by replacing `download: False` in a yaml config file with `download: True`.

### 3.1 Download the datasets
```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

### 3.2 Untar and extract files
```shell
# Go to the root of this repository
mkdir ~/dataset/ -p
mv VOCtrainval_11-May-2012.tar ~/dataset/
cd ~/dataset/
tar -xvf ILSVRC2012_img_val.tar
cd ../../
```
