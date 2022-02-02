# SC2 Benchmark: Supervised Compression for Split Computing

## Installation
```shell
pip install sc2bench
```

### Virtual environments
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


### Optional Software
If you want to use BPG, 
```shell
bash script/software/install_bpg.sh
```

The script will place the encoder and decoder in `~/software/`

## Datasets
See instructions [here](script#datasets)


## Codec-based feature compression
```shell
# JPEG
python script/task/image_classification.py -test_only --config configs/ilsvrc2012/feature_compression/jpeg-resnet50.yaml
# WebP
python script/task/image_classification.py -test_only --config configs/ilsvrc2012/feature_compression/webp-resnet50.yaml
```


## Analysis

### Trade-off

```shell
python3 tradeoff_plotter.py --input resource/analysis/offload_cost_vs_model_acc_size.tsv --x param_count --y top1_acc --models mnasnet_100  pnasnet5large mobilenetv3_large_100 inception_v4 inception_v3

python3 tradeoff_plotter.py --input resource/analysis/offload_cost_vs_model_acc_size.tsv --x jpeg_file_size --y top1_acc --models mnasnet_100  pnasnet5large mobilenetv3_large_100 inception_v4 inception_v3
```
