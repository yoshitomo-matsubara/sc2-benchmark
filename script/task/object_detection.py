import argparse
import builtins as __builtin__
import datetime
import json
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate
from torchdistill.common import file_util, module_util, yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed, \
    import_dependencies
from torchdistill.core.distillation import get_distillation_box
from torchdistill.core.training import get_training_box
from torchdistill.datasets.util import build_data_loader
from torchdistill.misc.log import set_basic_log_config, setup_log_file, SmoothedValue, MetricLogger
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN

from coco.dataset import get_coco_api_from_dataset
from coco.eval import CocoEvaluator
from sc2bench.analysis import check_if_analyzable
from sc2bench.common.config_util import overwrite_config
from sc2bench.models.detection.base import check_if_updatable_detection_model
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model

logger = def_logger.getChild(__name__)
torch.multiprocessing.set_sharing_strategy('file_system')


def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for object detection tasks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--json', help='json string to overwrite config')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--run_log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--iou_types', nargs='+', help='IoU types for evaluation '
                                                       '(the first IoU type is used for checkpoint selection)')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    parser.add_argument('-no_dp_eval', action='store_true',
                        help='perform evaluation without DistributedDataParallel/DataParallel')
    parser.add_argument('-log_config', action='store_true', help='log config')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    return parser


def load_model(model_config, device):
    if 'detection_model' not in model_config:
        return load_detection_model(model_config, device)
    return get_wrapped_detection_model(model_config, device)


def train_one_epoch(training_box, aux_module, bottleneck_updated, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    uses_aux_loss = aux_module is not None and not bottleneck_updated
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        start_time = time.time()
        supp_dict = default_collate(supp_dict)
        loss = training_box.forward_process(sample_batch, targets, supp_dict)
        aux_loss = None
        if uses_aux_loss:
            aux_loss = aux_module.aux_loss()
            aux_loss.backward()

        training_box.post_forward_process(loss)
        batch_size = len(sample_batch)
        if uses_aux_loss:
            metric_logger.update(loss=loss.item(), aux_loss=aux_loss.item(),
                                 lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])

        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError('The training loop was broken due to loss = {}'.format(loss))


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, DistributedDataParallel):
        model_without_ddp = model.module

    iou_type_list = ['bbox']
    if isinstance(model_without_ddp, MaskRCNN):
        iou_type_list.append('segm')
    if isinstance(model_without_ddp, KeypointRCNN):
        iou_type_list.append('keypoints')
    return iou_type_list


def log_info(*args, **kwargs):
    force = kwargs.pop('force', False)
    if is_main_process() or force:
        logger.info(*args, **kwargs)


@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, iou_types, device, device_ids, distributed, no_dp_eval=False,
             log_freq=1000, title=None, header='Test:'):
    model = model_wo_ddp.to(device)
    if distributed and not no_dp_eval:
        model = DistributedDataParallel(model, device_ids=device_ids)
    elif device.type.startswith('cuda') and not no_dp_eval:
        model = DataParallel(model, device_ids=device_ids)
    elif hasattr(model, 'use_cpu4compression'):
        model.use_cpu4compression()

    if title is not None:
        logger.info(title)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    # Replace built-in print function with logger.info to log summary printed by pycocotools
    builtin_print = __builtin__.print
    __builtin__.print = log_info

    cpu_device = torch.device('cpu')
    model.eval()
    analyzable = check_if_analyzable(model_wo_ddp)
    metric_logger = MetricLogger(delimiter='  ')
    coco = get_coco_api_from_dataset(data_loader.dataset)
    if iou_types is None or (isinstance(iou_types, (list, tuple)) and len(iou_types) == 0):
        iou_types = get_iou_types(model)

    coco_evaluator = CocoEvaluator(coco, iou_types)
    for sample_batch, targets in metric_logger.log_every(data_loader, log_freq, header):
        sample_batch = list(image.to(device) for image in sample_batch)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(sample_batch)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_stats_str = 'Averaged stats: {}'.format(metric_logger)
    logger.info(avg_stats_str)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # Revert print function
    __builtin__.print = builtin_print

    torch.set_num_threads(n_threads)
    if analyzable and model_wo_ddp.activated_analysis:
        model_wo_ddp.summarize()
    return coco_evaluator


def train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path, device, device_ids, distributed, config, args):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_map = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(src_ckpt_file_path):
        best_val_map, _ = load_ckpt(src_ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    iou_types = args.iou_types
    val_iou_type = iou_types[0] if isinstance(iou_types, (list, tuple)) and len(iou_types) > 0 else 'bbox'
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    aux_module = student_model_without_ddp.get_aux_module() \
        if check_if_updatable_detection_model(student_model_without_ddp) else None
    epoch_to_update = train_config.get('epoch_to_update', None)
    bottleneck_updated = False
    no_dp_eval = args.no_dp_eval
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_epoch_process(epoch=epoch)
        if epoch_to_update is not None and epoch_to_update <= epoch and not bottleneck_updated:
            logger.info('Updating entropy bottleneck')
            student_model_without_ddp.update()
            bottleneck_updated = True

        train_one_epoch(training_box, aux_module, bottleneck_updated, device, epoch, log_freq)
        val_coco_evaluator =\
            evaluate(student_model, training_box.val_data_loader, iou_types, device, device_ids, distributed,
                     no_dp_eval=no_dp_eval, log_freq=log_freq, header='Validation:')
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        val_map = val_coco_evaluator.coco_eval[val_iou_type].stats[0]
        if val_map > best_val_map and is_main_process():
            logger.info('Best mAP ({}): {:.4f} -> {:.4f}'.format(val_iou_type, best_val_map, val_map))
            logger.info('Updating ckpt at {}'.format(dst_ckpt_file_path))
            best_val_map = val_map
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler, best_val_map, args, dst_ckpt_file_path)
        training_box.post_epoch_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def main(args):
    set_basic_log_config()
    log_file_path = args.run_log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, world_size, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    logger.info(args)
    cudnn.benchmark = True
    cudnn.deterministic = True
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    if args.json is not None:
        logger.info('Overwriting config')
        overwrite_config(config, json.loads(args.json))

    import_dependencies(config.get('dependencies', None))
    device = torch.device(args.device)
    dataset_dict = config['datasets']
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model = load_model(teacher_model_config, device) if teacher_model_config is not None else None
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    src_ckpt_file_path = student_model_config.get('src_ckpt', None)
    dst_ckpt_file_path = student_model_config.get('dst_ckpt', None)
    student_model = load_model(student_model_config, device)
    if args.log_config:
        logger.info(config)

    student_model_without_ddp =\
        student_model.module if module_util.check_if_wrapped(student_model) else student_model
    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, src_ckpt_file_path, dst_ckpt_file_path,
              device, device_ids, distributed, config, args)

    load_ckpt(dst_ckpt_file_path, model=student_model_without_ddp, strict=True)
    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                         test_data_loader_config, distributed)
    log_freq = test_config.get('log_freq', 1000)
    no_dp_eval = args.no_dp_eval
    iou_types = args.iou_types
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, iou_types, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                 log_freq=log_freq, title='[Teacher: {}]'.format(teacher_model_config['key']))

    if check_if_updatable_detection_model(student_model):
        student_model.update()

    if check_if_analyzable(student_model):
        student_model.activate_analysis()
    evaluate(student_model, test_data_loader, iou_types, device, device_ids, distributed, no_dp_eval=no_dp_eval,
             log_freq=log_freq, title='[Student: {}]'.format(student_model_config['key']))


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
