from collections import OrderedDict

import torch
from compressai.models import CompressionModel
from timm.models import resnest, regnet, vision_transformer_hybrid
from torchdistill.datasets.util import build_transform
from torchdistill.models.registry import register_model_class, register_model_func
from torchvision import models
from torchvision.ops import misc as misc_nn_ops

from torchdistill.common.main_util import load_ckpt
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


def check_if_updatable(model):
    return isinstance(model, UpdatableBackbone)


class FeatureExtractionBackbone(UpdatableBackbone):
    # Referred to the IntermediateLayerGetter implementation at https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
    def __init__(self, model, return_layer_dict, analyzer_configs, analyzes_after_compress=False,
                 analyzable_layer_key=None):
        if not set(return_layer_dict).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layer_dict are not present in model')

        super().__init__(analyzer_configs)
        org_return_layer_dict = return_layer_dict
        return_layer_dict = {str(k): str(v) for k, v in return_layer_dict.items()}
        layer_dict = OrderedDict()
        for name, module in model.named_children():
            layer_dict[name] = module
            if name in return_layer_dict:
                return_layer_dict.pop(name)
            # Once all the return layers are extracted, the remaining layers are no longer used, thus pruned
            if len(return_layer_dict) == 0:
                break

        for key, module in layer_dict.items():
            self.add_module(key, module)

        self.return_layer_dict = org_return_layer_dict
        self.analyzable_layer_key = analyzable_layer_key
        self.analyzes_after_compress = analyzes_after_compress

    def forward(self, x):
        out = OrderedDict()
        for module_key, module in self.named_children():
            if module_key == self.analyzable_layer_key and self.bottleneck_updated and not self.training:
                x = module.encode(x)
                if self.analyzes_after_compress:
                    self.analyze(x)
                x = module.decode(**x)
            else:
                x = module(x)

            if module_key in self.return_layer_dict:
                out_name = self.return_layer_dict[module_key]
                out[out_name] = x
        return out

    def check_if_updatable(self, strict=True):
        if self.analyzable_layer_key is None or self.analyzable_layer_key not in self._modules \
                or not isinstance(self._modules[self.analyzable_layer_key], CompressionModel):
            return False
        return True

    def update(self):
        if self.analyzable_layer_key is None:
            return

        if not self.check_if_updatable() and isinstance(self._modules[self.analyzable_layer_key], CompressionModel):
            raise KeyError(f'`analyzable_layer_key` ({self.analyzable_layer_key}) does not '
                           f'exist in {self}')
        else:
            self._modules[self.analyzable_layer_key].update()
        self.bottleneck_updated = True

    def get_aux_module(self, **kwargs):
        if self.analyzable_layer_key is None:
            return None
        return self._modules[self.analyzable_layer_key] if self.check_if_updatable() else None


class SplittableResNet(UpdatableBackbone):
    # Referred to the ResNet implementation at https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
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
                entropy_bottleneck_state_dict[key.replace('bottleneck_layer.', '', 1)] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.bottleneck_layer if isinstance(self.bottleneck_layer, CompressionModel) else None


class SplittableRegNet(UpdatableBackbone):
    # Referred to the RegNet implementation at https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py
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
                entropy_bottleneck_state_dict[key.replace('bottleneck_layer.', '', 1)] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.bottleneck_layer if isinstance(self.bottleneck_layer, CompressionModel) else None


class SplittableHybridViT(UpdatableBackbone):
    # Referred to Hybrid ViT implementation at https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, bottleneck_layer, hybrid_vit_model, num_pruned_stages=1, skips_head=True,
                 pre_transform_params=None, analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.pre_transform = build_transform(pre_transform_params)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.bottleneck_layer = bottleneck_layer
        self.patch_embed_pruned_stages = hybrid_vit_model.patch_embed.backbone.stages[num_pruned_stages:]
        self.patch_embed_norm = hybrid_vit_model.patch_embed.backbone.norm
        self.patch_embed_head = hybrid_vit_model.patch_embed.backbone.head
        self.patch_embed_proj = hybrid_vit_model.patch_embed.proj
        self.cls_token = hybrid_vit_model.cls_token
        self.pos_embed = hybrid_vit_model.pos_embed
        self.pos_drop = hybrid_vit_model.pos_drop
        self.blocks = hybrid_vit_model.blocks
        self.norm = hybrid_vit_model.norm
        self.pre_logits = hybrid_vit_model.pre_logits
        self.head = None if skips_head else hybrid_vit_model.head

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

        x = self.patch_embed_pruned_stages(x)
        x = self.patch_embed_norm(x)
        x = self.patch_embed_head(x)
        x = self.patch_embed_proj(x).flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
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
                entropy_bottleneck_state_dict[key.replace('bottleneck_layer.', '', 1)] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.bottleneck_layer.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.bottleneck_layer if isinstance(self.bottleneck_layer, CompressionModel) else None


@register_backbone_func
def splittable_resnet(bottleneck_config, resnet_name='resnet50', inplanes=None, skips_avgpool=True, skips_fc=True,
                      pre_transform_params=None, analysis_config=None, org_model_ckpt_file_path_or_url=None,
                      org_ckpt_strict=True, **resnet_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    if resnet_kwargs.pop('norm_layer', '') == 'FrozenBatchNorm2d':
        resnet_model = models.__dict__[resnet_name](norm_layer=misc_nn_ops.FrozenBatchNorm2d, **resnet_kwargs)
    else:
        resnet_model = models.__dict__[resnet_name](**resnet_kwargs)

    if org_model_ckpt_file_path_or_url is not None:
        load_ckpt(org_model_ckpt_file_path_or_url, model=resnet_model, strict=org_ckpt_strict)
    return SplittableResNet(bottleneck_layer, resnet_model, inplanes, skips_avgpool, skips_fc,
                            pre_transform_params, analysis_config)


@register_backbone_func
def splittable_resnest(bottleneck_config, resnest_name='resnest50d', inplanes=None, skips_avgpool=True, skips_fc=True,
                       pre_transform_params=None, analysis_config=None, org_model_ckpt_file_path_or_url=None,
                       org_ckpt_strict=True, **resnest_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    resnest_model = resnest.__dict__[resnest_name](**resnest_kwargs)
    if org_model_ckpt_file_path_or_url is not None:
        load_ckpt(org_model_ckpt_file_path_or_url, model=resnest_model, strict=org_ckpt_strict)
    return SplittableResNet(bottleneck_layer, resnest_model, inplanes, skips_avgpool, skips_fc,
                            pre_transform_params, analysis_config)


@register_backbone_func
def splittable_regnet(bottleneck_config, regnet_name='regnety_064', inplanes=None, skips_head=True,
                      pre_transform_params=None, analysis_config=None, org_model_ckpt_file_path_or_url=None,
                      org_ckpt_strict=True, **regnet_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    regnet_model = regnet.__dict__[regnet_name](**regnet_kwargs)
    if org_model_ckpt_file_path_or_url is not None:
        load_ckpt(org_model_ckpt_file_path_or_url, model=regnet_model, strict=org_ckpt_strict)
    return SplittableRegNet(bottleneck_layer, regnet_model, inplanes, skips_head, pre_transform_params, analysis_config)


@register_backbone_func
def splittable_hybrid_vit(bottleneck_config, hybrid_vit_name='vit_small_r26_s32_224', num_pruned_stages=1,
                          skips_head=True, pre_transform_params=None, analysis_config=None,
                          org_model_ckpt_file_path_or_url=None, org_ckpt_strict=True, **hybrid_vit_kwargs):
    bottleneck_layer = get_layer(bottleneck_config['name'], **bottleneck_config['params'])
    hybrid_vit_model = vision_transformer_hybrid.__dict__[hybrid_vit_name](**hybrid_vit_kwargs)
    if org_model_ckpt_file_path_or_url is not None:
        load_ckpt(org_model_ckpt_file_path_or_url, model=hybrid_vit_model, strict=org_ckpt_strict)
    return SplittableHybridViT(bottleneck_layer, hybrid_vit_model, num_pruned_stages, skips_head,
                               pre_transform_params, analysis_config)


def get_backbone(cls_or_func_name, **kwargs):
    if cls_or_func_name in BACKBONE_CLASS_DICT:
        return BACKBONE_CLASS_DICT[cls_or_func_name](**kwargs)
    elif cls_or_func_name in BACKBONE_FUNC_DICT:
        return BACKBONE_FUNC_DICT[cls_or_func_name](**kwargs)
    return None
