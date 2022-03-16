# SC2: Supervised Compression for Split Computing
This is the official repository of `sc2bench` package and ["SC2: Supervised Compression for Split Computing"](#Citation).

## Installation
```shell
pip install sc2bench
```

### Virtual Environments
For pipenv users,
```shell
pipenv install --python 3.8
pipenv install -e "."
```

For conda users,
```shell
conda env create -f environment.yaml
conda activate sc2-benchmark
pip install -e "."
```


## Datasets
See instructions [here](script#datasets)

## Checkpoints
You can download our checkpoints including trained model weights [here](https://drive.google.com/file/d/1bIfKm_3X_imiWXHkqxy0rFZfafxBRaB4/view?usp=sharing).  
Unzip the downloaded zip file under ./, then there will be ./resource/ckpt/.

## Citation
[[Preprint]()]
```bibtex

```

## Note
For measuring data size per sample precisely, it is important to keep test batch size of 1 when testing.  
E.g., some baseline modules may expect larger batch size if you have multiple GPUs.  
Then, add `CUDA_VISIBLE_DEVICES=0` before your execution command (e.g., `sh`, `bash`, `python`) 
so that you can force the script to use one GPU (use GPU: 0 in this case).

For instance, an input compression experiment using factorized prior (pretrained input compression model) 
and ResNet-50 (pretrained classifier)
```shell
CUDA_VISIBLE_DEVICES=0 sh script/neural_input_compression/ilsvrc2012-image_classification.sh factorized_prior-resnet50 8
```

## Baselines: Input Compression
- [Codec-based input compression](https://github.com/yoshitomo-matsubara/sc2-benchmark/tree/main/script/codec_input_compression): JPEG, WebP, BPG
- [Neural input compression](https://github.com/yoshitomo-matsubara/sc2-benchmark/tree/main/script/neural_input_compression): Factorized Prior, Scale Hyperprior, Mean-scale Hyperprior, and Joint Autoregressive Hierarchical Prior

Each **README.md** gives instructions to run the baseline experiments.


## Codec-based Feature Compression
```shell
# JPEG
python script/task/image_classification.py -test_only --config configs/ilsvrc2012/feature_compression/jpeg-resnet50.yaml
# WebP
python script/task/image_classification.py -test_only --config configs/ilsvrc2012/feature_compression/webp-resnet50.yaml
```
