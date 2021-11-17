import numpy as np
from torch import nn
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_binary_object_size

logger = def_logger.getChild(__name__)
ANALYZER_CLASS_DICT = dict()


def register_analysis_class(cls):
    """
    Args:
        cls (class): analyzer module to be registered.

    Returns:
        cls (class): registered analyzer module.
    """
    ANALYZER_CLASS_DICT[cls.__name__] = cls
    return cls


class AnalyzableModule(nn.Module):
    """
    Base module to analyze and summarize the wrapped modules and intermediate representations.
    Args:
        analyzer_configs (list of dicts): list of configurations to instantiate analyzers
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


class BaseAnalyzer(object):
    """
    Base analyzer to analyze and summarize the wrapped modules and intermediate representations.
    """
    def analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def summarize(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


@register_analysis_class
class FileSizeAnalyzer(BaseAnalyzer):
    """
    Analyzer to measure file size of compressed object in the designated unit
    Args:
        unit (str): unit of data size in bytes (`B`, `KB`, `MB`)
        kwargs (dict): keyword arguments
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        self.unit = unit
        self.unit_size = self.UNIT_DICT[unit]
        self.kwargs = kwargs
        self.file_size_list = list()

    def analyze(self, compressed_obj):
        file_size = get_binary_object_size(compressed_obj, unit_size=self.unit_size)
        self.file_size_list.append(file_size)

    def summarize(self):
        file_sizes = np.array(self.file_size_list)
        logger.info('Bottleneck size [{}]: mean {} std {} for {} samples'.format(self.unit, file_sizes.mean(),
                                                                                 file_sizes.std(), len(file_sizes)))

    def clear(self):
        self.file_size_list.clear()


@register_analysis_class
class FileSizeAccumulator(FileSizeAnalyzer):
    """
    Accumulator to store pre-computed file size in the designated unit
    Args:
        unit (str): unit of data size in bytes (`B`, `KB`, `MB`)
        kwargs (dict): keyword arguments
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        super().__init__(unit=unit, **kwargs)

    def analyze(self, file_size):
        self.file_size_list.append(file_size / self.unit_size)


def get_analyzer(cls_name, **kwargs):
    """
    Args:
        cls_name (str): module class name.
        kwargs (dict): keyword arguments.

    Returns:
        BaseAnalyzer or None: analyzer module that is instance of `BaseAnalyzer` if found. None otherwise.
    """
    if cls_name not in ANALYZER_CLASS_DICT:
        return None
    return ANALYZER_CLASS_DICT[cls_name](**kwargs)


def check_if_analyzable(module):
    """
    Args:
        module (torch.nn.Module): PyTorch module to be checked.

    Returns:
        bool: True if model is instance of `AnalyzableModule`. False otherwise.
    """
    return isinstance(module, AnalyzableModule)
