from collections import OrderedDict

import torch
from torch import nn
from torchdistill.common.main_util import load_ckpt
from torchdistill.datasets.util import build_transform
from torchdistill.models.util import redesign_model

from .backbone import UpdatableBackbone
from .layer import EntropyBottleneckLayer
from .registry import get_compression_model, load_classification_model
from ..analysis import AnalyzableModule

WRAPPER_CLASS_DICT = dict()


def register_wrapper_class(cls):
    """
    Args:
        cls (class): wrapper module to be registered.

    Returns:
        cls (class): registered wrapper module.
    """
    WRAPPER_CLASS_DICT[cls.__name__] = cls
    return cls


@register_wrapper_class
class CodecInputCompressionClassifier(AnalyzableModule):
    """
    Wrapper module for codec input compression model followed by classifier.
    Args:
        classification_model (nn.Module): classification model
        codec_params (dict): keyword configurations for transform sequence for codec
        post_transform_params (dict): keyword configurations for transform sequence after compression model
        analysis_config (dict): configuration for analysis
    """
    def __init__(self, classification_model, device, codec_params=None,
                 post_transform_params=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.codec_encoder_decoder = build_transform(codec_params)
        self.device = device
        self.classification_model = classification_model
        self.post_transform = build_transform(post_transform_params)

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
    Wrapper module for neural input compression model followed by classifier.
    Args:
        classification_model (nn.Module): classification model
        pre_transform_params (dict): keyword configurations for transform sequence for input data
        compression_model (nn.Module): neural input compression model
        post_transform_params (dict): keyword configurations for transform sequence after compression model
        analysis_config (dict): configuration for analysis
    """
    def __init__(self, classification_model, pre_transform_params=None, compression_model=None,
                 post_transform_params=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.analyzes_after_pre_transform = analysis_config.get('analyzes_after_pre_transform', False)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.pre_transform = build_transform(pre_transform_params)
        self.compression_model = compression_model
        self.classification_model = classification_model
        self.post_transform = build_transform(post_transform_params)

    def forward(self, x):
        """
        Args:
            x (list of PIL Images or Tensor): input sample.

        Returns:
            Tensor: output tensor from self.classification_model.
        """

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
    Wrapper module for codec feature compression model injected to a classifier.
    Args:
        classification_model (nn.Module): classification model
        encoder_config (dict): keyword configurations to design an encoder from modules in classification_model
        codec_params (dict): keyword configurations for transform sequence for codec
        decoder_config (dict): keyword configurations to design a decoder from modules in classification_model
        classifier_config (dict): keyword configurations to design a classifier from modules in classification_model
        post_transform_params (dict): keyword configurations for transform sequence after compression model
        analysis_config (dict): configuration for analysis
    """
    def __init__(self, classification_model, device, encoder_config=None, codec_params=None, decoder_config=None,
                 classifier_config=None, post_transform_params=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.codec_encoder_decoder = build_transform(codec_params)
        self.device = device
        self.encoder = redesign_model(classification_model, encoder_config, model_label='encoder')
        self.decoder = redesign_model(classification_model, decoder_config, model_label='decoder')
        self.classifier = redesign_model(classification_model, classifier_config, model_label='classification')
        self.post_transform = build_transform(post_transform_params)

    def forward(self, x):
        """
        Args:
            x (Tensor): input sample.

        Returns:
            Tensor: output tensor from self.classifier.
        """
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
    Wrapper module for entropic compression model injected to a classifier.
    Args:
        classification_model (nn.Module): classification model
        encoder_config (dict): keyword configurations to design an encoder from modules in classification_model
        compression_model_params (dict): keyword configurations for CompressionModel in compressai
        decoder_config (dict): keyword configurations to design a decoder from modules in classification_model
        classifier_config (dict): keyword configurations to design a classifier from modules in classification_model
        analysis_config (dict): configuration for analysis
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
        """
        Args:
            x (Tensor): input sample.

        Returns:
            Tensor: output tensor from self.classifier.
        """
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
        entropy_bottleneck_state_dict = OrderedDict()
        for key in list(state_dict.keys()):
            if key.startswith('entropy_bottleneck.'):
                entropy_bottleneck_state_dict[key.replace('entropy_bottleneck.', '', 1)] = state_dict.pop(key)

        super().load_state_dict(state_dict, strict=False)
        self.entropy_bottleneck.load_state_dict(entropy_bottleneck_state_dict)

    def get_aux_module(self, **kwargs):
        return self.entropy_bottleneck


def wrap_model(wrapper_model_name, model, compressor, **kwargs):
    """
    Args:
        wrapper_model_name (str): wrapper model key in wrapper model register.
        model (nn.Module): model to be wrapped.
        compressor (nn.Module): compressor to be wrapped.
        **kwargs (dict): keyword arguments to instantiate a wrapper object.

    Returns:
        nn.Module: a wrapper module.
    """
    if wrapper_model_name not in WRAPPER_CLASS_DICT:
        raise ValueError('wrapper_model_name `{}` is not expected'.format(wrapper_model_name))
    return WRAPPER_CLASS_DICT[wrapper_model_name](model, compressor=compressor, **kwargs)


def get_wrapped_model(wrapper_model_config, task, device, distributed):
    """
    Args:
        wrapper_model_config (dict): wrapper model configuration.
        task (str): task category ('classification', 'object_detection', or 'semantic_segmentation').
        device (device): torch device.
        distributed (bool): uses distributed training model.

    Returns:
        nn.Module: a wrapped module.
    """
    wrapper_model_name = wrapper_model_config['name']
    if wrapper_model_name not in WRAPPER_CLASS_DICT:
        raise ValueError('wrapper_model_name `{}` is not expected'.format(wrapper_model_name))

    if task == 'classification':
        compression_model_config = wrapper_model_config.get('compression_model', None)
        compression_model = get_compression_model(compression_model_config, device)
        classification_model_config = wrapper_model_config['classification_model']
        model = load_classification_model(classification_model_config, device, distributed)
    else:
        raise ValueError(f'task `{task}` is not expected')

    wrapped_model = WRAPPER_CLASS_DICT[wrapper_model_name](model, compression_model=compression_model, device=device,
                                                           **wrapper_model_config['params'])
    ckpt_file_path = wrapper_model_config.get('ckpt', None)
    if ckpt_file_path is not None:
        load_ckpt(ckpt_file_path, model=wrapped_model, strict=False)
    return wrapped_model
