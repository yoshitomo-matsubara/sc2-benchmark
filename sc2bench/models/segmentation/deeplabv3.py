import torch
from torch.hub import load_state_dict_from_url
from torchdistill.common.main_util import load_ckpt
from torchvision.models.segmentation.deeplabv3 import model_urls, DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from .base import BaseSegmentationModel
from .registry import register_segmentation_model_func
from ..backbone import FeatureExtractionBackbone
from ..registry import load_classification_model


def create_deeplabv3(backbone, num_input_channels=2048, uses_aux=False, num_aux_channels=1024, num_classes=21):
    """
    Builds DeepLabv3 model using a given updatable backbone model.

    :param backbone: backbone model (usually a classification model)
    :type backbone: nn.Module
    :param num_input_channels: number of input channels for classification head
    :type num_input_channels: int
    :param uses_aux: If True, add an auxiliary branch
    :type uses_aux: bool
    :param num_aux_channels: number of input channels for auxiliary classification head
    :type num_aux_channels: int
    :param num_classes: number of output classes of the model (including the background)
    :type num_classes: int
    """
    aux_classifier = None
    if uses_aux:
        aux_classifier = FCNHead(num_aux_channels, num_classes)

    classifier = DeepLabHead(num_input_channels, num_classes)
    return BaseSegmentationModel(backbone, classifier, aux_classifier)


@register_segmentation_model_func
def deeplabv3_model(backbone_config, pretrained=True, pretrained_backbone_name=None, progress=True,
                    num_input_channels=2048, uses_aux=False, num_aux_channels=1024, return_layer_dict=None,
                    num_classes=21, analysis_config=None, analyzable_layer_key=None, start_ckpt_file_path=None,
                    **kwargs):
    """
    Builds DeepLabv3 model using a given updatable backbone model.


    :param backbone_config: backbone configuration
    :type backbone_config: dict
    :param pretrained: if True, returns a model pre-trained on COCO train2017 (torchvision)
    :type pretrained: bool
    :param pretrained_backbone_name: pretrained backbone name such as
            `'resnet50'`, `'resnet101'`, and `'mobilenet_v3_large'`
    :type pretrained_backbone_name: str
    :param progress: if True, displays a progress bar of the download to stderr
    :type progress: bool
    :param num_input_channels: number of input channels for classification head
    :type num_input_channels: int
    :param uses_aux: If True, add an auxiliary branch
    :type uses_aux: bool
    :param num_aux_channels: number of input channels for auxiliary classification head
    :type num_aux_channels: int
    :param return_layer_dict: mapping from name of module to return its output to a specified key
    :type return_layer_dict: dict
    :param num_classes: number of output classes of the model (including the background)
    :type num_classes: int
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    :param analyzable_layer_key: key of analyzable layer
    :type analyzable_layer_key: str
    :param start_ckpt_file_path: checkpoint file path to be loaded for the built DeepLabv3 model
    :type start_ckpt_file_path: str or None
    """
    if analysis_config is None:
        analysis_config = dict()

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
    if pretrained and pretrained_backbone_name in ('resnet50', 'resnet101'):
        state_dict = \
            load_state_dict_from_url(model_urls['deeplabv3_{}_coco'.format(pretrained_backbone_name)],
                                     progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if start_ckpt_file_path is not None:
        load_ckpt(start_ckpt_file_path, model=model, strict=False)
    return model
