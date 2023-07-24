import numpy as np
import torch
from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_binary_object_size

logger = def_logger.getChild(__name__)
ANALYZER_CLASS_DICT = dict()


def register_analysis_class(cls):
    """
    Registers an analyzer class.

    :param cls: analyzer class to be registered
    :type cls: class
    :return: cls: registered analyzer class
    :rtype: cls: class
    """
    ANALYZER_CLASS_DICT[cls.__name__] = cls
    return cls


class AnalyzableModule(nn.Module):
    """
    A base module to analyze and summarize the wrapped modules and intermediate representations.

    :param analyzer_configs: list of analysis configurations
    :type analyzer_configs: list[dict] or None
    """
    def __init__(self, analyzer_configs=None):
        if analyzer_configs is None:
            analyzer_configs = list()

        super().__init__()
        self.analyzers = [get_analyzer(analyzer_config['type'], **analyzer_config['params'])
                          for analyzer_config in analyzer_configs]
        self.activated_analysis = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def activate_analysis(self):
        """
        Makes internal analyzers ready to run.
        """
        self.activated_analysis = True

    def deactivate_analysis(self):
        """
        Turns internal analyzers off.
        """
        self.activated_analysis = False

    def analyze(self, compressed_obj):
        """
        Analyzes a compressed object using internal analyzers.

        :param compressed_obj: compressed object
        :type compressed_obj: Any
        """
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj)

    def summarize(self):
        """
        Shows each of internal analyzers' summary of results.
        """
        for analyzer in self.analyzers:
            analyzer.summarize()

    def clear_analysis(self):
        """
        Clears each of internal analyzers' results.
        """
        for analyzer in self.analyzers:
            analyzer.clear()


class BaseAnalyzer(object):
    """
    A base analyzer to analyze and summarize the wrapped modules and intermediate representations.
    """
    def analyze(self, *args, **kwargs):
        """
        Analyzes a compressed object.
        """
        raise NotImplementedError()

    def summarize(self):
        """
        Shows the summary of results.

        This should be overridden by all subclasses.
        """
        raise NotImplementedError()

    def clear(self):
        """
        Clears the results.

        This should be overridden by all subclasses.
        """
        raise NotImplementedError()


@register_analysis_class
class FileSizeAnalyzer(BaseAnalyzer):
    """
    An analyzer to measure file size of compressed object in the designated unit.

    :param unit: unit of data size in bytes ('B', 'KB', 'MB')
    :type unit: str
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        self.unit = unit
        self.unit_size = self.UNIT_DICT[unit]
        self.kwargs = kwargs
        self.file_size_list = list()

    def analyze(self, compressed_obj):
        """
        Computes and appends binary object size of the compressed object.

        :param compressed_obj: compressed object
        :type compressed_obj: Any
        """
        file_size = get_binary_object_size(compressed_obj, unit_size=self.unit_size)
        self.file_size_list.append(file_size)

    def summarize(self):
        """
        Computes and shows mean and std of the stored file sizes and the number of samples .
        """
        file_sizes = np.array(self.file_size_list)
        logger.info('Bottleneck size [{}]: mean {} std {} for {} samples'.format(self.unit, file_sizes.mean(),
                                                                                 file_sizes.std(), len(file_sizes)))

    def clear(self):
        """
        Clears the file size list.
        """
        self.file_size_list.clear()


@register_analysis_class
class FileSizeAccumulator(FileSizeAnalyzer):
    """
    An accumulator to store pre-computed file size in the designated unit.

    :param unit: unit of data size in bytes ('B', 'KB', 'MB')
    :type unit: str
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        super().__init__(unit=unit, **kwargs)

    def analyze(self, file_size):
        """
        Appends a file size.

        :param file_size: pre-computed file size
        :type file_size: int or float
        """
        self.file_size_list.append(file_size / self.unit_size)


def get_analyzer(cls_name, **kwargs):
    """
    Gets an analyzer module.

    :param cls_name: analyzer class name
    :type cls_name: str
    :param kwargs: kwargs for the analyzer class
    :type kwargs: dict
    :return: analyzer module
    :rtype: BaseAnalyzer or None
    """
    if cls_name not in ANALYZER_CLASS_DICT:
        return None
    return ANALYZER_CLASS_DICT[cls_name](**kwargs)


def check_if_analyzable(module):
    """
    Checks if a module is an instance of `AnalyzableModule`.

    :param module: module
    :type module: Any
    :return: True if the module is an instance of `AnalyzableModule`. False otherwise
    :rtype: bool
    """
    return isinstance(module, AnalyzableModule)


def analyze_model_size(model, encoder_paths=None, additional_rest_paths=None, ignores_dtype_error=True):
    """
    Approximates numbers of bits used for parameters of the whole model, encoder, and the rest of the model.

    :param model: model
    :type model: nn.Module
    :param encoder_paths: list of module paths for the model to be considered as part of encoder's parameters
    :type encoder_paths: list[str] or None
    :param additional_rest_paths: list of additional rest module paths whose parameters should be shared with encoder
                            e.g., module path of entropy bottleneck in the model if applied
    :type additional_rest_paths: list[str] or None
    :param ignores_dtype_error: if False, raise an error when any unexpected dtypes are found
    :type ignores_dtype_error: bool
    :return: model size (sum of param x num_bits) with three keys: model (whole model), encoder, and the rest
    :rtype: dict
    """
    model_size = 0
    encoder_size = 0
    rest_size = 0
    if encoder_paths is None:
        encoder_paths = list()

    if additional_rest_paths is None:
        additional_rest_paths = list()

    encoder_path_set = set(encoder_paths)
    additional_rest_path_set = set(additional_rest_paths)
    for k, v in model.state_dict().items():
        dim = v.dim()
        param_count = 1 if dim == 0 else np.prod(v.size())
        v_dtype = v.dtype
        if v_dtype in (torch.int64, torch.float64):
            num_bits = 64
        elif v_dtype in (torch.int32, torch.float32):
            num_bits = 32
        elif v_dtype in (torch.int16, torch.float16, torch.bfloat16):
            num_bits = 16
        elif v_dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
            num_bits = 8
        elif v_dtype == torch.bool:
            num_bits = 2
        else:
            error_message = f'For {k}, dtype `{v_dtype}` is not expected'
            if ignores_dtype_error:
                print(error_message)
                continue
            else:
                raise TypeError(error_message)

        param_size = num_bits * param_count
        model_size += param_size
        match_flag = False
        for encoder_path in encoder_path_set:
            if k.startswith(encoder_path):
                encoder_size += param_size
                if k in additional_rest_path_set:
                    rest_size += param_size
                match_flag = True
                break

        if not match_flag:
            rest_size += param_size
    return {'model': model_size, 'encoder': encoder_size, 'rest': rest_size}
