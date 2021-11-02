import argparse
import json
import os

import torch
from compressai.zoo import models as compression_model_dict
from torch.backends import cudnn
from torch.nn import DataParallel, functional
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common import file_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt
from torchdistill.datasets import util
from torchdistill.datasets.util import build_transform
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, MetricLogger
from torchdistill.models.official import get_image_classification_model
from torchdistill.models.registry import get_model

import sc2bench

logger = def_logger.getChild(__name__)
cudnn.benchmark = True
cudnn.deterministic = True
torch.multiprocessing.set_sharing_strategy('file_system')


def get_argparser():
    parser = argparse.ArgumentParser(description='Input compression for image classification')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--json', help='json string to overwrite config')
    parser.add_argument('--comp_device', default='cuda', help='device for compression model')
    parser.add_argument('--class_device', default='cuda', help='device for classification model')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('-log_config', action='store_true', help='log config')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def overwrite_config(org_config, sub_config):
    for sub_key, sub_value in sub_config.items():
        if sub_key in org_config:
            if isinstance(sub_value, dict):
                overwrite_config(org_config[sub_key], sub_value)
            else:
                org_config[sub_key] = sub_value
        else:
            org_config[sub_key] = sub_value


def load_compression_model(model_config, device):
    if model_config is None:
        return None, None

    model_name = model_config['name']
    model_kwargs = model_config['params']
    if model_name in compression_model_dict:
        model = compression_model_dict[model_name](**model_kwargs)
    else:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_name, repo_or_dir, **model_kwargs)

    ckpt_file_path = model_config['ckpt']
    load_ckpt(ckpt_file_path, model=model, strict=None)
    logger.info('Updating compression model')
    model.update()
    post_transform = \
        build_transform(model_config['post_transform_params']) if 'post_transform_params' in model_config else None
    return model.to(device), post_transform


def load_classification_model(model_config, device, distributed):
    model = get_image_classification_model(model_config, distributed, False)
    if model is None:
        repo_or_dir = model_config.get('repo_or_dir', None)
        model = get_model(model_config['name'], repo_or_dir, **model_config['params'])

    model_ckpt_file_path = os.path.expanduser(model_config['ckpt'])
    load_ckpt(model_ckpt_file_path, model=model, strict=False)
    return model.to(device)


def pad_input(x):
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    pad_tuple = (padding_left, padding_right, padding_top, padding_bottom)
    return functional.pad(x, pad_tuple, mode='constant', value=0), pad_tuple


def unpad_output(x, pad_tuple):
    neg_pad_tuple = tuple(-p for p in pad_tuple)
    return functional.pad(x, neg_pad_tuple)


@torch.inference_mode()
def evaluate(compression_model, post_transform, classification_model, data_loader,
             comp_device, class_device, device_ids, distributed, log_freq=1000, title=None, header='Test:'):
    if distributed:
        classification_model = DistributedDataParallel(classification_model, device_ids=device_ids)
    elif class_device.type.startswith('cuda'):
        classification_model = DataParallel(classification_model, device_ids=device_ids)

    if title is not None:
        logger.info(title)

    if compression_model is not None:
        compression_model.eval()

    classification_model.eval()
    metric_logger = MetricLogger(delimiter='  ')
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        batch_size = image.shape[0]
        if compression_model is not None:
            image = image.to(comp_device)
            padded_image, pad_tuple = pad_input(image)
            enc_output = compression_model.compress(padded_image)
            compressed_data_size = file_util.get_binary_object_size(enc_output)
            metric_logger.meters['data_size'].update(compressed_data_size, n=batch_size)
            dec_output = compression_model.decompress(enc_output['strings'], enc_output['shape'])
            image = unpad_output(dec_output['x_hat'], pad_tuple)
            if post_transform is not None:
                image = post_transform(image)

        image, target = image.to(class_device, non_blocking=True), target.to(class_device, non_blocking=True)
        output = classification_model(image)
        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    if compression_model is None:
        logger.info(' * Acc@1: {:.4f}\tAcc@5: {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    else:
        data_size = metric_logger.data_size.global_avg
        logger.info(' * Acc@1: {:.4f}\tAcc@5: {:.4f}\tData size: {:.4f} [KB]\n'.format(top1_accuracy,
                                                                                    top5_accuracy, data_size))
    return metric_logger.acc1.global_avg


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    logger.info(f'sc2bench ver. {sc2bench.__version__}')
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    if args.json is not None:
        overwrite_config(config, json.loads(args.json))

    if args.log_config:
        logger.info(config)

    comp_device = torch.device(args.comp_device)
    class_device = torch.device(args.class_device)
    dataset_dict = util.get_all_datasets(config['datasets'])
    models_config = config['models']
    compression_model, post_transform = load_compression_model(models_config.get('compression', None), comp_device)
    classifier_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    classification_model = load_classification_model(classifier_config, class_device, distributed)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    evaluate(compression_model, post_transform, classification_model, test_data_loader, comp_device,
             class_device, device_ids, distributed, title='[Student: {}]'.format(classifier_config['name']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
