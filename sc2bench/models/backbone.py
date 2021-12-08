from collections import OrderedDict

import torch
from timm.models import resnest, regnet
from torchdistill.datasets.util import build_transform
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models
from torchvision.ops import misc as misc_nn_ops

from .layer import get_layer
from ..analysis import AnalyzableModule

BACKBONE_CLASS_DICT = dict()
BACKBONE_FUNC_DICT = dict()


def register_backbone_class(cls):
    BACKBONE_CLASS_DICT[cls.__name__] = cls
    register_model_class(cls)
    return cls


def register_backbone_func(func):
    BACKBONE_FUNC_DICT[func.__name__] = func
    register_model_func(func)
    return func


class UpdatableBackbone(AnalyzableModule):
    def __init__(self, analyzer_configs=None):
        super().__init__(analyzer_configs)
        self.bottleneck_updated = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, **kwargs):
        raise NotImplementedError()

    def get_aux_module(self, **kwargs):
        raise NotImplementedError()


class SplittableResNet(UpdatableBackbone):
    def __init__(self, bottleneck_layer, resnet_model, inplanes=None, skips_avgpool=True, skips_fc=True,
                 pre_transform_params=None, analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.pre_transform = build_transform(pre_transform_params)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.bottleneck_layer = bottleneck_layer
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = None if skips_avgpool \
            else resnet_model.global_pool if hasattr(resnet_model, 'global_pool') else resnet_model.avgpool
        self.fc = None if skips_fc else resnet_model.fc
        self.inplanes = resnet_model.inplanes if inplanes is None else inplanes

    def forward(self, x):
        if self.pre_transform is not None:
            x = self.pre_transform(x)

        if self.bottleneck_updated and not self.training:
            x = self.bottleneck_layer.encode(x)
            if self.analyzes_after_compress:
                self.analyze(x)
            x = self.bottleneck_layer.decode(**x)
        else:
            x = self.bottleneck_layer(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.avgpool is None:
            return x

        x = self.avgpool(x)
        if self.fc is None:
            return x

        x = torch.flatten(x, 1)
        return self.fc(x)

    def update(self):
        self.bottleneck_layer.update()
        self.bottleneck_updated = True

    def load_state_dict(self, state_dict, **kwargs):
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith('bottleneck_layer.'):
                entropy_bottleneck_state_dict[key.replace('bottleneck_layer.', '')] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.bottleneck_layer


class SplittableRegNet(UpdatableBackbone):
    def __init__(self, bottleneck_layer, regnet_model, inplanes=None, skips_head=True,
                 pre_transform_params=None, analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.pre_transform = build_transform(pre_transform_params)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.bottleneck_layer = bottleneck_layer
        self.s2 = regnet_model.s2
        self.s3 = regnet_model.s3
        self.s4 = regnet_model.s4
        self.head = None if skips_head else regnet_model.head
        self.inplanes = inplanes

    def forward(self, x):
        if self.pre_transform is not None:
            x = self.pre_transform(x)

        if self.bottleneck_updated and not self.training:
            x = self.bottleneck_layer.encode(x)
            if self.analyzes_after_compress:
                self.analyze(x)
            x = self.bottleneck_layer.decode(**x)
        else:
            x = self.bottleneck_layer(x)

        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        if self.head is None:
            return x
        return self.head(x)

    def update(self):
        self.bottleneck_layer.update()
        self.bottleneck_updated = True

    def load_state_dict(self, state_dict, **kwargs):
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith('bottleneck_layer.'):
                entropy_bottleneck_state_dict[key.replace('bottleneck_layer.', '')] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.bottleneck_layer


@register_backbone_func
def splittable_resnet(bottleneck_config, resnet_name='resnet50', inplanes=None, skips_avgpool=True, skips_fc=True,
                      pre_transform_params=None, analysis_config=None, **resnet_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    if resnet_kwargs.pop('norm_layer', '') == 'FrozenBatchNorm2d':
        resnet_model = models.__dict__[resnet_name](norm_layer=misc_nn_ops.FrozenBatchNorm2d, **resnet_kwargs)
    else:
        resnet_model = models.__dict__[resnet_name](**resnet_kwargs)
    return SplittableResNet(bottleneck_layer, resnet_model, inplanes, skips_avgpool, skips_fc,
                            pre_transform_params, analysis_config)


@register_backbone_func
def splittable_resnest(bottleneck_config, resnest_name='resnest50d', inplanes=None, skips_avgpool=True, skips_fc=True,
                       pre_transform_params=None, analysis_config=None, **resnest_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    resnest_model = resnest.__dict__[resnest_name](**resnest_kwargs)
    return SplittableResNet(bottleneck_layer, resnest_model, inplanes, skips_avgpool, skips_fc,
                            pre_transform_params, analysis_config)


@register_backbone_func
def splittable_regnet(bottleneck_config, reget_name='regnety_064', inplanes=None, skips_head=True,
                      pre_transform_params=None, analysis_config=None, **resnest_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    regnet_model = regnet.__dict__[reget_name](**resnest_kwargs)
    return SplittableRegNet(bottleneck_layer, regnet_model, inplanes, skips_head,
                            pre_transform_params, analysis_config)


def get_backbone(cls_or_func_name, **kwargs):
    if cls_or_func_name in BACKBONE_CLASS_DICT:
        return BACKBONE_CLASS_DICT[cls_or_func_name](**kwargs)
    elif cls_or_func_name in BACKBONE_FUNC_DICT:
        return BACKBONE_FUNC_DICT[cls_or_func_name](**kwargs)
    return None


def check_if_updatable(model):
    return isinstance(model, UpdatableBackbone)
