dependencies = ['torch', 'torchvision', 'compressai', 'timm']

from sc2bench.models.backbone import splittable_resnet, splittable_densenet, splittable_inception_v3


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
    skips_avgpool = 'avgpool' in short_module_name_set
    skips_fc = 'fc' in short_module_name_set
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
    skips_avgpool = 'avgpool' in short_module_name_set
    skips_fc = 'fc' in short_module_name_set
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
    skips_avgpool = 'avgpool' in short_module_name_set
    skips_fc = 'fc' in short_module_name_set
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
                                   skips_avgpool=False, skips_fc=False, **kwargs)
