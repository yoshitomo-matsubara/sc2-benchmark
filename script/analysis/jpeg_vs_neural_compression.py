import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

TARGET_PAIRS = (
    ('jpeg.tsv', 'JPEG compression'),
    ('factorized_prior.tsv', 'Factorized Prior'),
    ('scale_hyperprior.tsv', 'Scale Hyperprior'),
    ('mean_scale_hyperprior.tsv', 'Mean-Scale Hyperprior')
)


def get_args():
    parser = argparse.ArgumentParser(description='JPEG vs Neural compression')
    parser.add_argument('--input', required=True, help='input dir path')
    return parser.parse_args()


def plot_lines(input_file_path, label):
    if not os.path.exists(input_file_path):
        print(f'`{input_file_path}` was not found')
        return

    data_mat = np.loadtxt(input_file_path, delimiter='\t').transpose()
    x_values = data_mat[1]
    y_values = data_mat[2]
    plt.plot(x_values, y_values, '--s', label=label)


def main(args):
    input_dir_path = args.input
    for input_file_name, label in TARGET_PAIRS:
        input_file_path = os.path.join(input_dir_path, input_file_name)
        plot_lines(input_file_path, label)

    plt.xlabel('File Size [KB]')
    plt.ylabel('Top-1 Accuracy [%]')
    plt.title('EfficientNet-L2 (NS)')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(get_args())
