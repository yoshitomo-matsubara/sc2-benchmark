import argparse
import math
import os
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdistill.datasets.transform import CustomCompose, CustomRandomResize
from torchdistill.datasets.util import load_coco_dataset, build_transform
from torchvision.datasets import ImageFolder, VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import transforms

from sc2bench.transforms.codec import PillowImageModule, BpgModule, VtmModule
from sc2bench.transforms.misc import ClearTargetTransform

torch.multiprocessing.set_sharing_strategy('file_system')


def get_argparser():
    parser = argparse.ArgumentParser(description='Compressed file size by codec '
                                                 'for ImageNet and COCO segmentation datasets')
    parser.add_argument('--dataset', required=True, choices=['imagenet', 'coco_segment', 'pascal_segment'],
                        help='ckpt dir path')
    parser.add_argument('--img_size', type=int, default=224, help='input image size')
    parser.add_argument('--crop_pct', type=float, default=0.875, help='crop rate')
    parser.add_argument('--min_quality', type=int, default=0, help='min quality for codec compression')
    parser.add_argument('--quality_step', type=int, default=10, help='step size in quality for codec compression')
    parser.add_argument('--max_quality', type=int, default=100, help='max quality for codec compression')
    parser.add_argument('--interpolation', default='bicubic', help='crop rate')
    parser.add_argument('--format', default='JPEG', help='codec format')
    parser.add_argument('-highest_only', action='store_true', help='compute compressed size with highest quality only')
    return parser


def get_codec_module(codec_format, quality):
    if codec_format == 'BPG':
        return BpgModule(encoder_path='~/software/libbpg-0.9.8/bpgenc', decoder_path='~/software/libbpg-0.9.8/bpgdec',
                         quality=quality, returns_file_size=True)
    elif codec_format == 'VTM':
        return VtmModule(encoder_path='~/software/VVCSoftware_VTM/bin/EncoderAppStatic',
                         decoder_path='~/software/VVCSoftware_VTM/bin/DecoderAppStatic',
                         config_path='~/software/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg',
                         color_mode='ycbcr', quality=quality, returns_file_size=True)
    return PillowImageModule(returns_file_size=True, format=codec_format, quality=quality)


def get_codec_module_kwargs(codec_format, quality):
    if codec_format == 'BPG':
        return {'encoder_path': '~/software/libbpg-0.9.8/bpgenc', 'decoder_path': '~/software/libbpg-0.9.8/bpgdec',
                'quality': quality, 'returns_file_size': True}
    elif codec_format == 'VTM':
        return {'encoder_path': '~/software/VVCSoftware_VTM/bin/EncoderAppStatic',
                'decoder_path': '~/software/VVCSoftware_VTM/bin/DecoderAppStatic',
                'config_path': '~/software/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg',
                'color_mode': 'ycbcr',
                'quality': quality, 'returns_file_size': True}
    return {'returns_file_size': True, 'format': codec_format, 'quality': quality}


def compute_compressed_file_size_with_transform(dataset, codec_format, quality, batch_size=1):
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=False)
    file_size_list = list()
    for img_sizes, _ in data_loader:
        file_size_list.extend(img_sizes)

    file_sizes = np.array(file_size_list) / 1024
    print('{} quality: {}, File size [KB]: {} ± {} for {} samples'.format(codec_format, quality, file_sizes.mean(),
                                                                          file_sizes.std(), len(file_sizes)))


def compute_compressed_file_size_for_imagenet_dataset(codec_format, min_quality, step_size, max_quality,
                                                      img_size=224, crop_pct=0.875, interpolation='bicubic',
                                                      highest_only=False):
    # Refer to https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
    # for img_size, crop_pct, and method variables
    crop_size = int(math.floor(img_size / crop_pct))
    if interpolation == 'bicubic':
        pil_interpolation = InterpolationMode.BICUBIC
    elif interpolation == 'lanczos':
        pil_interpolation = InterpolationMode.LANCZOS
    elif interpolation == 'hamming':
        pil_interpolation = InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        pil_interpolation = InterpolationMode.BILINEAR

    print('img_size: {}\tcrop_pct: {}\tinterpolation: {}'.format(img_size, crop_pct, interpolation))
    root_dir_path = os.path.expanduser('~/dataset/ilsvrc2012/val')
    sub_transform_list = [
        transforms.Resize(crop_size, pil_interpolation),
        transforms.CenterCrop(img_size)
    ]
    for i, quality in enumerate(range(max_quality, min_quality, -step_size)):
        codec_transform = get_codec_module(codec_format, quality)
        transform = transforms.Compose(sub_transform_list + [codec_transform])
        dataset = ImageFolder(root=root_dir_path, transform=transform)
        compute_compressed_file_size_with_transform(dataset, codec_format, quality, batch_size=1000)
        if highest_only and i == 0:
            return


def compute_compressed_file_size(split_config, is_segment, transform, codec_format, quality):
    dataset = load_coco_dataset(split_config['images'], split_config['annotations'],
                                split_config['annotated_only'], split_config.get('random_horizontal_flip', None),
                                is_segment, transform, split_config.get('quality', None))
    data_loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    file_size_list = list()
    for img_sizes, _ in data_loader:
        file_size_list.extend(img_sizes)

    file_sizes = np.array(file_size_list) / 1024
    print('{} quality: {}, File size [KB]: {} ± {} for {} samples'.format(codec_format, quality, file_sizes.mean(),
                                                                          file_sizes.std(), len(file_sizes)))


def compute_compressed_file_size_for_cocosegment_dataset(codec_format, min_quality, step_size, max_quality):
    for quality in range(max_quality, min_quality, -step_size):
        split_config = {
            'images': '~/dataset/coco2017/val2017',
            'annotations': '~/dataset/coco2017/annotations/instances_val2017.json',
            'annotated_only': False,
            'is_segment': True,
            'transforms_params': [
                {'type': 'CustomRandomResize', 'params': {'min_size': 520, 'max_size': 520}},
                get_codec_module_kwargs(codec_format, quality),
                {'type': 'ClearTargetTransform', 'params': {}},
            ]
        }
        is_segment = split_config.get('is_segment', False)
        compose_cls = CustomCompose if is_segment else None
        transform = build_transform(split_config.get('transforms_params', None), compose_cls=compose_cls)
        compute_compressed_file_size(split_config, is_segment, transform, codec_format, quality)


def compute_compressed_file_size_for_pascalsegment_dataset(codec_format, min_quality, step_size, max_quality):
    for quality in range(max_quality, min_quality, -step_size):
        transform = CustomCompose([
            CustomRandomResize(min_size=512, max_size=512),
            get_codec_module(codec_format, quality),
            ClearTargetTransform()
        ])
        dataset = VOCSegmentation(root=os.path.expanduser('~/dataset/'), image_set='val', year='2012',
                                  transforms=transform)
        compute_compressed_file_size_with_transform(dataset, codec_format, quality)


def main(args):
    print(args)
    codec_format = args.format
    min_quality, step_size, max_quality, = args.min_quality, args.quality_step, args.max_quality
    if args.dataset == 'imagenet':
        compute_compressed_file_size_for_imagenet_dataset(codec_format, min_quality, step_size, max_quality,
                                                          args.img_size, args.crop_pct, args.interpolation,
                                                          args.highest_only)
    elif args.dataset == 'coco_segment':
        compute_compressed_file_size_for_cocosegment_dataset(codec_format, min_quality, step_size, max_quality)
    else:
        compute_compressed_file_size_for_pascalsegment_dataset(codec_format, min_quality, step_size, max_quality)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
