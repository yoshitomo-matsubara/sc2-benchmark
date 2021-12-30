import torch
from torch.hub import load_state_dict_from_url
from torchdistill.common.main_util import load_ckpt
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.segmentation import model_urls

from .base import BaseSegmentationModel
from .registry import register_segmentation_model_func
from ..backbone import FeatureExtractionBackbone
from ..registry import load_classification_model


def create_deeplabv3(backbone_model, num_input_channels=2048, uses_aux=False, num_aux_channels=1024, num_classes=21):
    aux_classifier = None
    if uses_aux:
        aux_classifier = FCNHead(num_aux_channels, num_classes)

    classifier = DeepLabHead(num_input_channels, num_classes)
    return BaseSegmentationModel(backbone_model, classifier, aux_classifier)


@register_segmentation_model_func
def custom_deeplabv3(backbone_config, pretrained=True, progress=True, num_input_channels=2048,
                     uses_aux=False, num_aux_channels=1024, return_layer_dict=None, analysis_config=None,
                     analyzable_layer_key=None, num_classes=21, start_ckpt_file_path=None, **kwargs):
    if analysis_config is None:
        analysis_config = dict()

    backbone_name = backbone_config['name']
    if return_layer_dict is None:
        return_layer_dict = {'layer4': 'out'}
        if uses_aux:
            return_layer_dict['layer3'] = 'aux'

    backbone = load_classification_model(backbone_config, torch.device('cpu'), False, strict=False)
    backbone_model = \
        FeatureExtractionBackbone(backbone, return_layer_dict, analysis_config.get('analyzer_configs', list()),
                                  analysis_config.get('analyzes_after_compress', False),
                                  analyzable_layer_key=analyzable_layer_key)
    model = create_deeplabv3(backbone_model, num_input_channels=num_input_channels,
                             uses_aux=uses_aux, num_aux_channels=num_aux_channels, num_classes=num_classes)
    if pretrained and (backbone_name.endswith('resnet50') or backbone_name.endswith('resnet101')):
        resnet_name = 'resnet50' if backbone_name.endswith('resnet50') else 'resnet101'
        state_dict = \
            load_state_dict_from_url(model_urls['deeplabv3_{}_coco'.format(resnet_name)], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if start_ckpt_file_path is not None:
        load_ckpt(start_ckpt_file_path, model=model, strict=False)
    return model
