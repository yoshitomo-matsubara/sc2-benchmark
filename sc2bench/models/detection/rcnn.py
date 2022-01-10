import torch
from torch.hub import load_state_dict_from_url
from torchdistill.common.main_util import load_ckpt
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.faster_rcnn import FasterRCNN, model_urls as faster_rcnn_model_urls
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .base import UpdatableDetectionModel, UpdatableBackboneWithFPN
from .registry import register_detection_model_func
from ..backbone import check_if_updatable
from ..registry import load_classification_model
from ...analysis import check_if_analyzable


class BaseRCNN(GeneralizedRCNN, UpdatableDetectionModel):
    # Referred to https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
    def __init__(self, rcnn_model, analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()

        UpdatableDetectionModel.__init__(self, analysis_config.get('analyzer_configs', list()))
        GeneralizedRCNN.__init__(self, rcnn_model.backbone, rcnn_model.rpn, rcnn_model.roi_heads, rcnn_model.transform)

    def update(self, **kwargs):
        if not check_if_updatable(self.backbone.body):
            raise KeyError(f'`backbone` {type(self)} is not updatable')
        self.backbone.body.update()

    def get_aux_module(self, **kwargs):
        return self.backbone.body.get_aux_module()

    def activate_analysis(self):
        self.activated_analysis = True
        if check_if_analyzable(self.backbone.body):
            self.backbone.body.activate_analysis()

    def deactivate_analysis(self):
        self.activated_analysis = False
        self.backbone.body.deactivate_analysis()
        if check_if_analyzable(self.backbone.body):
            self.backbone.body.deactivate_analysis()

    def analyze(self, compressed_obj):
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj)
        if check_if_analyzable(self.backbone.body):
            self.backbone.body.analyze(compressed_obj)

    def summarize(self):
        for analyzer in self.analyzers:
            analyzer.summarize()
        if check_if_analyzable(self.backbone.body):
            self.backbone.body.summarize()

    def clear_analysis(self):
        for analyzer in self.analyzers:
            analyzer.clear()
        if check_if_analyzable(self.backbone.body):
            self.backbone.body.clear_analysis()


def create_faster_rcnn_fpn(backbone, extra_blocks=None, return_layer_dict=None, in_channels_list=None,
                           in_channels_stage2=None, out_channels=256, returned_layers=None, num_classes=91,
                           analysis_config=None, analyzable_layer_key=None, **kwargs):
    if analysis_config is None:
        analysis_config = dict()

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    if return_layer_dict is None:
        return_layer_dict = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    if in_channels_stage2 is None:
        in_channels_stage2 = backbone.inplanes // 8

    if in_channels_list is None:
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]

    backbone_with_fpn = \
        UpdatableBackboneWithFPN(backbone, return_layer_dict, in_channels_list, out_channels, extra_blocks=extra_blocks,
                                 analyzable_layer_key=analyzable_layer_key, **analysis_config)
    return FasterRCNN(backbone_with_fpn, num_classes, **kwargs)


def _process_torchvision_pretrained_weights(model, pretrained_backbone_name, progress):
    base_backbone_name = 'resnet50'
    if pretrained_backbone_name == 'mobilenet_v3_large_320':
        base_backbone_name = 'mobilenet_v3_large_320'
    elif pretrained_backbone_name == 'mobilenet_v3_large':
        base_backbone_name = 'mobilenet_v3_large'
    state_dict = \
        load_state_dict_from_url(faster_rcnn_model_urls['fasterrcnn_{}_fpn_coco'.format(base_backbone_name)],
                                 progress=progress)
    model.load_state_dict(state_dict, strict=False)
    if pretrained_backbone_name == 'resnet50':
        overwrite_eps(model, 0.0)


@register_detection_model_func
def faster_rcnn_model(backbone_config, pretrained=True, pretrained_backbone_name=None, progress=True,
                      backbone_fpn_kwargs=None, analysis_config=None, num_classes=91,
                      start_ckpt_file_path=None, **kwargs):
    if backbone_fpn_kwargs is None:
        backbone_fpn_kwargs = dict()

    if analysis_config is None:
        analysis_config = dict()

    backbone_config['params']['norm_layer'] = misc_nn_ops.FrozenBatchNorm2d
    backbone = load_classification_model(backbone_config, torch.device('cpu'), False, strict=False)

    faster_rcnn_model = create_faster_rcnn_fpn(backbone, num_classes=num_classes, **backbone_fpn_kwargs, **kwargs)
    model = BaseRCNN(faster_rcnn_model, analysis_config=analysis_config)
    if pretrained and pretrained_backbone_name in ('resnet50', 'mobilenet_v3_large_320', 'mobilenet_v3_large'):
        _process_torchvision_pretrained_weights(model, pretrained_backbone_name, progress)

    if start_ckpt_file_path is not None:
        load_ckpt(start_ckpt_file_path, model=model, strict=False)
    return model
