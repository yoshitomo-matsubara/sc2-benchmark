import argparse

import matplotlib.pyplot as plt
import pandas as pd


KEY_DICT = {
    'top1_acc': ('top1 acc', 'ImageNet Top-1 Accuracy [%]'),
    'param_count': ('#params [M]', 'Number of Parameters (Millions)'),
    'jpeg_file_size': ('jpeg file size [KB]', 'Offloading Cost (File Size [KB])')
}

SERIES_DICT = {
    'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
    'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
    'tf_efficientnet_ns': ['tf_efficientnet_b0_ns', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2_ns',
                           'tf_efficientnet_b3_ns', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5_ns',
                           'tf_efficientnet_b6_ns', 'tf_efficientnet_b7_ns'],
    'regnet': ['regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_160'],
    'resnest': ['resnest14d', 'resnest26d', 'resnest50d', 'resnest101e', 'resnest200e', 'resnest269e'],
}


def get_args():
    parser = argparse.ArgumentParser(description='Tradeoff plotter for ImageNet dataset')
    parser.add_argument('--input', required=True, help='input TSV file path')
    parser.add_argument('--x', required=True, choices=['top1_acc', 'param_count', 'jpeg_file_size'],
                        help='data key for x-axis')
    parser.add_argument('--y', required=True, choices=['top1_acc', 'param_count', 'jpeg_file_size'],
                        help='data key for y-axis')
    parser.add_argument('--models', nargs='+', help='model name(s)')
    return parser.parse_args()


def expand_model_keys(org_model_keys):
    model_key_list = list()
    for org_model_key in org_model_keys:
        if org_model_key in SERIES_DICT:
            model_key_list.extend(SERIES_DICT[org_model_key])
        else:
            model_key_list.append(org_model_key)
    return model_key_list


def rename_labels(org_model_names):
    model_name_list = list()
    for org_model_name in org_model_names:
        model_name = org_model_name
        if model_name.startswith('inception_v'):
            model_name = model_name.replace('inception_v', 'Inception-v')
        elif model_name == 'pnasnet5large':
            model_name = 'PNASNet-5 (Large)'
        elif model_name == 'mnasnet_100':
            model_name = 'MnasNet 1.0'
        elif model_name == 'mobilenetv3_large_100':
            model_name = 'MobileNetV3 large (1.0)'
        model_name_list.append(model_name)
    return model_name_list


def plot_lines(data_frame, x_key, y_key):
    # sub_data_frame = data_frame[data_frame['model name'].str.fullmatch('|'.join(SERIES_DICT['densenet']))]
    # x_values = sub_data_frame[x_key].tolist()
    # y_values = sub_data_frame[y_key].tolist()
    # plt.plot(x_values, y_values, ':')
    # plt.text(max(x_values) - 4, max(y_values) + 0.25, 'DenseNet')

    sub_data_frame = data_frame[data_frame['model name'].str.fullmatch('|'.join(SERIES_DICT['resnet']))]
    x_values = sub_data_frame[x_key].tolist()
    y_values = sub_data_frame[y_key].tolist()
    plt.plot(x_values, y_values, ':.')
    plt.text(max(x_values) - 4, max(y_values) + 0.25, 'ResNet')

    sub_data_frame = data_frame[data_frame['model name'].str.fullmatch('|'.join(SERIES_DICT['tf_efficientnet_ns']))]
    x_values = sub_data_frame[x_key].tolist()
    y_values = sub_data_frame[y_key].tolist()
    plt.plot(x_values, y_values, '-*')
    plt.text(max(x_values) - 4, max(y_values) + 0.25, 'EfficientNet (NS)')

    sub_data_frame = data_frame[data_frame['model name'].str.fullmatch('|'.join(SERIES_DICT['regnet']))]
    x_values = sub_data_frame[x_key].tolist()
    y_values = sub_data_frame[y_key].tolist()
    plt.plot(x_values, y_values, '-.^')
    plt.text(max(x_values) - 4, max(y_values) + 0.25, 'RegNet')

    sub_data_frame = data_frame[data_frame['model name'].str.fullmatch('|'.join(SERIES_DICT['resnest']))]
    x_values = sub_data_frame[x_key].tolist()
    y_values = sub_data_frame[y_key].tolist()
    plt.plot(x_values, y_values, '--s')
    plt.text(max(x_values) - 4, max(y_values) + 0.25, 'ResNeSt')


def main(args):
    data_frame = pd.read_csv(args.input, delimiter='\t')
    model_keys = expand_model_keys(args.models)
    sub_data_frame = data_frame[data_frame['model name'].str.fullmatch('|'.join(model_keys))]
    x_key, x_label = KEY_DICT[args.x]
    y_key, y_label = KEY_DICT[args.y]
    model_names = rename_labels(sub_data_frame['model name'].tolist())
    x_values = sub_data_frame[x_key].tolist()
    y_values = sub_data_frame[y_key].tolist()
    plt.scatter(x_values, y_values)
    for model_name, x, y in zip(model_names, x_values, y_values):
        plt.text(x, y, model_name)

    plot_lines(data_frame, x_key, y_key)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(get_args())
