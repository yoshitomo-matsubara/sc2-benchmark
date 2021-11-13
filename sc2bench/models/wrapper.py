from torch import nn
from torchdistill.datasets.util import build_transform
from .registry import get_compression_model, load_classification_model

from ..analysis import get_analyzer

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


class BaseWrapper(nn.Module):
    """
    Base wrapper module to analyze and summarize the wrapped modules and intermediate representations.
    Args:
        analyzer_configs (list of dicts): unit of data size in bytes (`B`, `KB`, `MB`)
    """
    def __init__(self, analyzer_configs):
        super().__init__()
        self.analyzers = [get_analyzer(analyzer_config['type'], **analyzer_config['params'])
                          for analyzer_config in analyzer_configs]
        self.activated_analysis = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def activate_analysis(self):
        self.activated_analysis = True

    def deactivate_analysis(self):
        self.activated_analysis = False

    def analyze(self, compressed_obj):
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj)

    def summarize(self):
        for analyzer in self.analyzers:
            analyzer.summarize()

    def clear_analysis(self):
        for analyzer in self.analyzers:
            analyzer.clear()


@register_wrapper_class
class InputCompressionClassifier(BaseWrapper):
    """
    Wrapper module for input compression model followed by classifier.
    Args:
        classification_model (nn.Module): classification model
        pre_transform_params (dict): keyword configurations for transform sequence for input data
        compression_model (nn.Module): input compression model
        post_transform_params (dict): keyword configurations for transform sequence after compression model
        analysis_config (dict): configuration for analysis
    """
    def __init__(self, classification_model, pre_transform_params=None, compression_model=None,
                 post_transform_params=None, analysis_config=None):
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
        compression_model_config = wrapper_model_config['compression_model']
        compression_model = get_compression_model(compression_model_config, device)
        classification_model_config = wrapper_model_config['classification_model']
        model = load_classification_model(classification_model_config, device, distributed)
    else:
        raise ValueError(f'task `{task}` is not expected')
    return WRAPPER_CLASS_DICT[wrapper_model_name](model, compression_model=compression_model,
                                                  **wrapper_model_config['params'])
