from collections import OrderedDict

import torch
from torch import nn
from torchdistill.common.main_util import load_ckpt
from torchdistill.models.util import redesign_model

from .backbone import UpdatableBackbone
from .layer import EntropyBottleneckLayer
from .registry import get_compression_model, load_classification_model
from ..analysis import AnalyzableModule

WRAPPER_CLASS_DICT = dict()


def register_wrapper_class(cls):
    """
    Registers a model wrapper class.

    :param cls: model wrapper to be registered
    :type cls: class
    :return: registered model wrapper class
    :rtype: class
    """
    WRAPPER_CLASS_DICT[cls.__name__] = cls
    return cls


@register_wrapper_class
class CodecInputCompressionClassifier(AnalyzableModule):
    """
    A wrapper module for codec input compression model followed by a classification model.

    :param classification_model: image classification model
    :type classification_model: nn.Module
    :param device: torch device
    :type device: torch.device or str
    :param codec_encoder_decoder: transform sequence configuration for codec
    :type codec_encoder_decoder: nn.Module or None
    :param post_transform: post-transform
    :type post_transform: nn.Module or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, classification_model, device, codec_encoder_decoder=None,
                 post_transform=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.codec_encoder_decoder = codec_encoder_decoder
        self.device = device
        self.classification_model = classification_model
        self.post_transform = post_transform

    def forward(self, x):
        """
        Args:
            x (list of PIL Images): input sample.

        Returns:
            Tensor: output tensor from self.classification_model.
        """

        tmp_list = list()
        for sub_x in x:
            if self.codec_encoder_decoder is not None:
                sub_x, file_size = self.codec_encoder_decoder(sub_x)
                if not self.training:
                    self.analyze(file_size)

            if self.post_transform is not None:
                sub_x = self.post_transform(sub_x)

            tmp_list.append(sub_x.unsqueeze(0))
        x = torch.hstack(tmp_list).to(self.device)
        return self.classification_model(x)


@register_wrapper_class
class NeuralInputCompressionClassifier(AnalyzableModule):
    """
    A wrapper module for neural input compression model followed by a classification model.

    :param classification_model: image classification model
    :type classification_model: nn.Module
    :param pre_transform: pre-transform
    :type pre_transform: nn.Module or None
    :param compression_model: compression model
    :type compression_model: nn.Module or None
    :param uses_cpu4compression_model: whether to use CPU instead of GPU for `compression_model`
    :type uses_cpu4compression_model: bool
    :param post_transform: post-transform
    :type post_transform: nn.Module or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, classification_model, pre_transform=None, compression_model=None,
                 uses_cpu4compression_model=False, post_transform=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.analyzes_after_pre_transform = analysis_config.get('analyzes_after_pre_transform', False)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.pre_transform = pre_transform
        self.compression_model = compression_model
        self.uses_cpu4compression_model = uses_cpu4compression_model
        self.classification_model = classification_model
        self.post_transform = post_transform

    def use_cpu4compression(self):
        """
        Changes the device of the compression model to CPU.
        """
        if self.uses_cpu4compression_model and self.compression_model is not None:
            self.compression_model = self.compression_model.cpu()

    def forward(self, x):
        if self.pre_transform is not None:
            x = self.pre_transform(x)
            if not self.training and self.analyzes_after_pre_transform:
                self.analyze(x)

        if self.compression_model is not None:
            compressed_obj = self.compression_model.compress(x)
            if not self.training and self.analyzes_after_compress:
                self.analyze(compressed_obj)
            x = self.compression_model.decompress(**compressed_obj)
            if isinstance(x, dict):
                x = x['x_hat']

        if self.post_transform is not None:
            x = self.post_transform(x)
        return self.classification_model(x)


@register_wrapper_class
class CodecFeatureCompressionClassifier(AnalyzableModule):
    """
    A wrapper module for codec feature compression model injected to a classification model.

    :param classification_model: image classification model
    :type classification_model: nn.Module
    :param device: torch device
    :type device: torch.device or str
    :param encoder_config: configuration to design an encoder using modules in classification_model
    :type encoder_config: dict or None
    :param codec_encoder_decoder: transform sequence configuration for codec
    :type codec_encoder_decoder: nn.Module or None
    :param decoder_config: configuration to design a decoder using modules in classification_model
    :type decoder_config: dict or None
    :param classifier_config: configuration to design a classifier using modules in classification_model
    :type classifier_config: dict or None
    :param post_transform: post-transform
    :type post_transform: nn.Module or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, classification_model, device, encoder_config=None, codec_encoder_decoder=None,
                 decoder_config=None, classifier_config=None, post_transform=None, analysis_config=None,
                 **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.codec_encoder_decoder = codec_encoder_decoder
        self.device = device

        self.encoder = nn.Identity() if encoder_config.get('ignored', False) \
            else redesign_model(classification_model, encoder_config, model_label='encoder')
        self.decoder = nn.Identity() if decoder_config.get('ignored', False) \
            else redesign_model(classification_model, decoder_config, model_label='decoder')
        self.classifier = redesign_model(classification_model, classifier_config, model_label='classification')
        self.post_transform = post_transform

    def forward(self, x):
        x = self.encoder(x)
        tmp_list = list()
        for sub_x in x:
            if self.codec_encoder_decoder is not None:
                sub_x, file_size = self.codec_encoder_decoder(sub_x)
                if not self.training:
                    self.analyze(file_size)

            if self.post_transform is not None:
                sub_x = self.post_transform(sub_x)
            tmp_list.append(sub_x.unsqueeze(0))

        x = torch.hstack(tmp_list).to(self.device)
        x = self.decoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@register_wrapper_class
class EntropicClassifier(UpdatableBackbone):
    """
    A wrapper module for entropic compression model injected to a classification model.

    :param classification_model: image classification model
    :type classification_model: nn.Module
    :param device: torch device
    :type device: torch.device or str
    :param encoder_config: configuration to design an encoder using modules in classification_model
    :type encoder_config: dict
    :param compression_model_params: kwargs for `EntropyBottleneckLayer` in `compressai`
    :type compression_model_params: dict
    :param decoder_config: configuration to design a decoder using modules in classification_model
    :type decoder_config: dict
    :param classifier_config: configuration to design a classifier using modules in classification_model
    :type classifier_config: dict
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, classification_model, encoder_config, compression_model_params, decoder_config,
                 classifier_config, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.entropy_bottleneck = EntropyBottleneckLayer(**compression_model_params)
        self.encoder = nn.Identity() if encoder_config.get('ignored', False) \
            else redesign_model(classification_model, encoder_config, model_label='encoder')
        self.decoder = nn.Identity() if decoder_config.get('ignored', False) \
            else redesign_model(classification_model, decoder_config, model_label='decoder')
        self.classifier = redesign_model(classification_model, classifier_config, model_label='classification')

    def forward(self, x):
        x = self.encoder(x)
        if self.bottleneck_updated and not self.training:
            x = self.entropy_bottleneck.compress(x)
            if self.analyzes_after_compress:
                self.analyze(x)
            x = self.entropy_bottleneck.decompress(**x)
        else:
            x, _ = self.entropy_bottleneck(x)

        x = self.decoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def update(self):
        self.entropy_bottleneck.update()
        self.bottleneck_updated = True

    def load_state_dict(self, state_dict, **kwargs):
        """
        Loads parameters for all the sub-modules except entropy_bottleneck and then entropy_bottleneck.

        :param state_dict: dict containing parameters and persistent buffers
        :type state_dict: dict
        """
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith('entropy_bottleneck.'):
                entropy_bottleneck_state_dict[key.replace('entropy_bottleneck.', '', 1)] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.entropy_bottleneck.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.entropy_bottleneck


@register_wrapper_class
class SplitClassifier(UpdatableBackbone):
    """
    A wrapper module for naively splitting a classification model.

    :param classification_model: image classification model
    :type classification_model: nn.Module
    :param encoder_config: configuration to design an encoder using modules in classification_model
    :type encoder_config: dict or None
    :param decoder_config: configuration to design a decoder using modules in classification_model
    :type decoder_config: dict or None
    :param classifier_config: configuration to design a classifier using modules in classification_model
    :type classifier_config: dict or None
    :param compressor_transform: compressor transform
    :type compressor_transform: nn.Module or None
    :param decompressor_transform: decompressor transform
    :type decompressor_transform: nn.Module or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, classification_model, encoder_config, decoder_config,
                 classifier_config, compressor_transform=None, decompressor_transform=None,
                 analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.compressor = compressor_transform
        self.decompressor = decompressor_transform
        self.encoder = nn.Identity() if encoder_config.get('ignored', False) \
            else redesign_model(classification_model, encoder_config, model_label='encoder')
        self.decoder = nn.Identity() if decoder_config.get('ignored', False) \
            else redesign_model(classification_model, decoder_config, model_label='decoder')
        self.classifier = redesign_model(classification_model, classifier_config, model_label='classification')

    def forward(self, x):
        x = self.encoder(x)
        if self.bottleneck_updated and not self.training:
            x = self.compressor(x)
            if self.analyzes_after_compress:
                self.analyze(x)
            x = self.decompressor(x)

        x = self.decoder(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def update(self):
        self.bottleneck_updated = True

    def get_aux_module(self, **kwargs):
        return None


def wrap_model(wrapper_model_name, model, compression_model, **kwargs):
    """
    Wraps a model and a compression model with a wrapper module.

    :param wrapper_model_name: wrapper model name
    :type wrapper_model_name: str
    :param model: model
    :type model: nn.Module
    :param compression_model: compression model
    :type compression_model: nn.Module
    :param kwargs: kwargs for the wrapper class or function to build the wrapper module
    :type kwargs: dict
    :return: wrapped model
    :rtype: nn.Module
    """
    if wrapper_model_name not in WRAPPER_CLASS_DICT:
        raise ValueError('wrapper_model_name `{}` is not expected'.format(wrapper_model_name))
    return WRAPPER_CLASS_DICT[wrapper_model_name](model, compression_model=compression_model, **kwargs)


def get_wrapped_classification_model(wrapper_model_config, device, distributed):
    """
    Gets a wrapped image classification model.

    :param wrapper_model_config: wrapper model configuration
    :type wrapper_model_config: dict
    :param device: torch device
    :type device: torch.device
    :param distributed: whether to use the model in distributed training mode
    :type distributed: bool
    :return: wrapped image classification model
    :rtype: nn.Module
    """
    wrapper_model_name = wrapper_model_config['key']
    if wrapper_model_name not in WRAPPER_CLASS_DICT:
        raise ValueError('wrapper_model_name `{}` is not expected'.format(wrapper_model_name))

    compression_model_config = wrapper_model_config.get('compression_model', None)
    compression_model = get_compression_model(compression_model_config, device)
    classification_model_config = wrapper_model_config['classification_model']
    model = load_classification_model(classification_model_config, device, distributed)
    wrapped_model = WRAPPER_CLASS_DICT[wrapper_model_name](model, compression_model=compression_model, device=device,
                                                           **wrapper_model_config['kwargs'])
    src_ckpt_file_path = wrapper_model_config.get('src_ckpt', None)
    if src_ckpt_file_path is not None:
        load_ckpt(src_ckpt_file_path, model=wrapped_model, strict=False)
    return wrapped_model
