dependencies = ['torch', 'torchvision', 'compressai', 'timm']

from torch.hub import load_state_dict_from_url
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops

from sc2bench.models.backbone import splittable_resnet, splittable_densenet, splittable_inception_v3
from sc2bench.models.layer import smaller_resnet_layer1_bottleneck, larger_resnet_layer1_bottleneck


def custom_resnet50(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                    short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    bottleneck_layer_config = {
        'key': 'larger_resnet_bottleneck',
        'kwargs': {
            'bottleneck_channel': bottleneck_channel,
            'bottleneck_idx': bottleneck_idx,
            'compressor_transform': compressor,
            'decompressor_transform': decompressor,
        }
    }
    short_module_name_set = set(short_module_names)
    skips_avgpool = 'avgpool' not in short_module_name_set
    skips_fc = 'fc' not in short_module_name_set
    return splittable_resnet(bottleneck_layer_config, resnet_name='resnet50',
                             skips_avgpool=skips_avgpool, skips_fc=skips_fc, short_module_names=short_module_names,
                             **kwargs)


def custom_resnet101(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    bottleneck_layer_config = {
        'key': 'larger_resnet_bottleneck',
        'kwargs': {
            'bottleneck_channel': bottleneck_channel,
            'bottleneck_idx': bottleneck_idx,
            'compressor_transform': compressor,
            'decompressor_transform': decompressor,
        }
    }
    short_module_name_set = set(short_module_names)
    skips_avgpool = 'avgpool' not in short_module_name_set
    skips_fc = 'fc' not in short_module_name_set
    return splittable_resnet(bottleneck_layer_config, resnet_name='resnet101',
                             skips_avgpool=skips_avgpool, skips_fc=skips_fc, short_module_names=short_module_names,
                             **kwargs)


def custom_resnet152(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    bottleneck_layer_config = {
        'key': 'larger_resnet_bottleneck',
        'kwargs': {
            'bottleneck_channel': bottleneck_channel,
            'bottleneck_idx': bottleneck_idx,
            'compressor_transform': compressor,
            'decompressor_transform': decompressor,
        }
    }
    short_module_name_set = set(short_module_names)
    skips_avgpool = 'avgpool' not in short_module_name_set
    skips_fc = 'fc' not in short_module_name_set
    return splittable_resnet(bottleneck_layer_config, resnet_name='resnet152',
                             skips_avgpool=skips_avgpool, skips_fc=skips_fc, short_module_names=short_module_names,
                             **kwargs)


def custom_densenet169(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                       short_feature_names=None, **kwargs):
    if short_feature_names is None:
        short_feature_names = ['denseblock3', 'transition3', 'denseblock4', 'norm5']

    bottleneck_layer_config = {
        'key': 'larger_densenet_bottleneck',
        'kwargs': {
            'bottleneck_channel': bottleneck_channel,
            'bottleneck_idx': bottleneck_idx,
            'compressor_transform': compressor,
            'decompressor_transform': decompressor,
        }
    }
    return splittable_densenet(bottleneck_layer_config, densenet_name='densenet169',
                               short_feature_names=short_feature_names, skips_avgpool=False, skips_classifier=False,
                               **kwargs)


def custom_densenet201(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                       short_feature_names=None, **kwargs):
    if short_feature_names is None:
        short_feature_names = ['denseblock3', 'transition3', 'denseblock4', 'norm5']

    bottleneck_layer_config = {
        'key': 'larger_densenet_bottleneck',
        'kwargs': {
            'bottleneck_channel': bottleneck_channel,
            'bottleneck_idx': bottleneck_idx,
            'compressor_transform': compressor,
            'decompressor_transform': decompressor,
        }
    }
    return splittable_densenet(bottleneck_layer_config, densenet_name='densenet201',
                               short_feature_names=short_feature_names, skips_avgpool=False, skips_classifier=False,
                               **kwargs)


def custom_inception_v3(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                       short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = [
            'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
            'Mixed_7a', 'Mixed_7b', 'Mixed_7c', 'fc'
        ]

    bottleneck_layer_config = {
        'key': 'inception_v3_bottleneck',
        'kwargs': {
            'bottleneck_channel': bottleneck_channel,
            'bottleneck_idx': bottleneck_idx,
            'compressor_transform': compressor,
            'decompressor_transform': decompressor,
        }
    }
    return splittable_inception_v3(bottleneck_layer_config, short_module_names=short_module_names,
                                   skips_avgpool=False, skips_dropout=False, skips_fc=False, **kwargs)


def custom_resnet_fpn_backbone(backbone_key, layer1, compressor=None, decompressor=None,
                               weights=None, trainable_backbone_layers=4, returned_layers=None,
                               norm_layer=misc_nn_ops.FrozenBatchNorm2d, **kwargs):
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    if backbone_key in {'custom_resnet18', 'custom_resnet34'}:
        layer1 = smaller_resnet_layer1_bottleneck(**layer1)
    elif backbone_key in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
        layer1 = larger_resnet_layer1_bottleneck(**layer1)

    prefix = 'custom_'
    start_idx = backbone_key.find(prefix) + len(prefix)
    org_backbone_key = backbone_key[start_idx:] if backbone_key.startswith(prefix) else backbone_key
    backbone = resnet.__dict__[org_backbone_key](
        weights=weights,
        norm_layer=norm_layer
    )
    if layer1 is not None:
        backbone.layer1 = layer1

    # select layers that won't be frozen
    assert 0 <= trainable_backbone_layers <= 6
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'bn1', 'conv1'][:trainable_backbone_layers]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def custom_fasterrcnn_resnet_fpn(backbone, weights=None, progress=True,
                                 num_classes=91, trainable_backbone_layers=3, **kwargs):
    backbone_key = backbone['key']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # don't freeze any layers if pretrained model or backbone is not used
    if weights is not None and 'trainable_backbone_layers' not in backbone_kwargs:
        backbone_kwargs['trainable_backbone_layers'] = 5

    backbone_model = custom_resnet_fpn_backbone(backbone_key, **backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, **kwargs)
    if weights is not None:
        state_dict = \
            load_state_dict_from_url(weights.url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def custom_maskrcnn_resnet_fpn(backbone, weights=None, progress=True,
                               num_classes=91, trainable_backbone_layers=3, **kwargs):
    backbone_key = backbone['key']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # don't freeze any layers if pretrained model or backbone is not used
    if weights is not None and 'trainable_backbone_layers' not in backbone_kwargs:
        backbone_kwargs['trainable_backbone_layers'] = 5

    backbone_model = custom_resnet_fpn_backbone(backbone_key, **backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    mask_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = MaskRCNN(backbone_model, num_classes, box_roi_pool=box_roi_pool, mask_roi_pool=mask_roi_pool, **kwargs)
    if weights is not None:
        state_dict = \
            load_state_dict_from_url(weights.url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def custom_keypointrcnn_resnet_fpn(backbone, weights=None, progress=True, num_classes=2, num_keypoints=17,
                                   trainable_backbone_layers=3, **kwargs):
    backbone_key = backbone['key']
    backbone_kwargs = backbone['kwargs']
    assert 0 <= trainable_backbone_layers <= 5
    # don't freeze any layers if pretrained model or backbone is not used
    if weights is not None and 'trainable_backbone_layers' not in backbone_kwargs:
        backbone_kwargs['trainable_backbone_layers'] = 5

    backbone_model = custom_resnet_fpn_backbone(backbone_key, **backbone_kwargs)
    num_feature_maps = len(backbone_model.body.return_layers)
    box_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=7, sampling_ratio=2)
    keypoint_roi_pool = None if num_feature_maps == 4 \
        else MultiScaleRoIAlign(featmap_names=[str(i) for i in range(num_feature_maps)],
                                output_size=14, sampling_ratio=2)
    model = KeypointRCNN(backbone_model, num_classes, num_keypoints=num_keypoints, box_roi_pool=box_roi_pool,
                         keypoint_roi_pool=keypoint_roi_pool, **kwargs)
    if weights is not None:
        state_dict = \
            load_state_dict_from_url(weights.url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

