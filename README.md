# SC2 Benchmark: Supervised Compression for Split Computing

python3 tradeoff_plotter.py --input resource/analysis/offload_cost_vs_model_acc_size.tsv --x param_count --y top1_acc --models mnasnet_100  pnasnet5large mobilenetv3_large_100 inception_v4 inception_v3

python3 tradeoff_plotter.py --input resource/analysis/offload_cost_vs_model_acc_size.tsv --x jpeg_file_size --y top1_acc --models mnasnet_100  pnasnet5large mobilenetv3_large_100 inception_v4 inception_v3
