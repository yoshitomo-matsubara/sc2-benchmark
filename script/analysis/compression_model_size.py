import matplotlib.pyplot as plt
import torchvision
from compressai.zoo import models
import numpy as np
from torch import nn
from torchdistill.common.module_util import count_params


def count_fp_params(model):
    num_encoder_params = count_params(nn.ModuleList([model.g_a, model.entropy_bottleneck]))
    num_decoder_params = count_params(nn.ModuleList([model.g_s, model.entropy_bottleneck]))
    return num_encoder_params, num_decoder_params


def count_hp_params(model):
    num_encoder_params = count_params(
        nn.ModuleList([model.g_a, model.h_a, model.h_s, model.entropy_bottleneck, model.gaussian_conditional]))
    num_decoder_params = count_params(
        nn.ModuleList([model.h_s, model.g_s, model.entropy_bottleneck, model.gaussian_conditional]))
    return num_encoder_params, num_decoder_params


def plot(x_ticks, encoder_counts, decoder_counts):
    mobilenet_v2_count = count_params(torchvision.models.mobilenet_v2())
    mobilenet_v3_count = count_params(torchvision.models.mobilenet_v3_large())
    plt.bar(x_ticks, encoder_counts, width=0.3, label='Encoder')
    plt.bar(x_ticks, decoder_counts, width=0.3, bottom=encoder_counts, label='Decoder')
    x_values = np.arange(-0.5, len(x_ticks))
    plt.plot(x_values, [mobilenet_v2_count for _ in range(len(x_ticks) + 1)], label='MobileNetV2', color='green')
    plt.plot(x_values, [mobilenet_v3_count for _ in range(len(x_ticks) + 1)], label='MobileNetV3', color='red')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Model Parameters')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    fp_encoder1_param_count, fp_decoder1_param_count = count_fp_params(models['bmshj2018-factorized'](quality=1))
    fp_encoder8_param_count, fp_decoder8_param_count = count_fp_params(models['bmshj2018-factorized'](quality=8))
    shp_encoder1_param_count, shp_decoder1_param_count = count_hp_params(models['bmshj2018-hyperprior'](quality=1))
    shp_encoder8_param_count, shp_decoder8_param_count = count_hp_params(models['bmshj2018-hyperprior'](quality=8))
    mshp_encoder1_param_count, mshp_decoder1_param_count = count_hp_params(models['mbt2018-mean'](quality=1))
    mshp_encoder8_param_count, mshp_decoder8_param_count = count_hp_params(models['mbt2018-mean'](quality=8))
    labels = ['Factorized Prior', 'Scale Hyperprior', 'Mean-Scale Hyperprior']

    small_encoder_counts = [fp_encoder1_param_count, shp_encoder1_param_count, mshp_encoder1_param_count]
    small_decoder_counts = [fp_decoder1_param_count, shp_decoder1_param_count, mshp_decoder1_param_count]
    small_x_ticks = [label + ' (small)' for label in labels]
    plot(small_x_ticks, small_encoder_counts, small_decoder_counts)

    large_encoder_counts = [fp_encoder8_param_count, shp_encoder8_param_count, mshp_encoder8_param_count]
    large_decoder_counts = [fp_decoder8_param_count, shp_decoder8_param_count, mshp_decoder8_param_count]
    large_x_ticks = [label + ' (large)' for label in labels]
    plot(large_x_ticks, large_encoder_counts, large_decoder_counts)

    encoder_count_list, decoder_count_list, x_tick_list = list(), list(), list()
    for small_encoder_count, large_encoder_count, small_decoder_count, large_decoder_count, small_x_tick, large_x_tick \
            in zip(small_encoder_counts, large_encoder_counts, small_decoder_counts, large_decoder_counts,
                   small_x_ticks, large_x_ticks):
        encoder_count_list.append(small_encoder_count)
        encoder_count_list.append(large_encoder_count)
        decoder_count_list.append(small_decoder_count)
        decoder_count_list.append(large_decoder_count)
        x_tick_list.append(small_x_tick)
        x_tick_list.append(large_x_tick)
    plot(x_tick_list, encoder_count_list, decoder_count_list)


if __name__ == '__main__':
    main()
