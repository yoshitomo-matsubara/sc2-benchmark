# SC2 Benchmark: Supervised Compression for Split Computing

## Setup
```shell
pipenv install --python 3.8
pipenv install -e "."
```

## Analysis

### Trade-off

```shell
python3 tradeoff_plotter.py --input resource/analysis/offload_cost_vs_model_acc_size.tsv --x param_count --y top1_acc --models mnasnet_100  pnasnet5large mobilenetv3_large_100 inception_v4 inception_v3

python3 tradeoff_plotter.py --input resource/analysis/offload_cost_vs_model_acc_size.tsv --x jpeg_file_size --y top1_acc --models mnasnet_100  pnasnet5large mobilenetv3_large_100 inception_v4 inception_v3
```

### Compressed file size

JPEG codec
```shell
pipenv run python script/analysis/codec_file_size.py --dataset imagenet --img_size 224 --crop_pct 0.875 --format JPEG
pipenv run python script/analysis/codec_file_size.py --dataset coco_segment --format JPEG
pipenv run python script/analysis/codec_file_size.py --dataset pascal_segment --format JPEG
```

WebP codec
```shell
pipenv run python script/analysis/codec_file_size.py --dataset imagenet --img_size 224 --crop_pct 0.875 --format WEBP
pipenv run python script/analysis/codec_file_size.py --dataset coco_segment --format WEBP
pipenv run python script/analysis/codec_file_size.py --dataset pascal_segment --format WEBP
```