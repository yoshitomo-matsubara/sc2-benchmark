from torch import nn
from torchdistill.datasets.util import build_transform

from sc2bench.analysis import get_analyzer

WRAPPER_CLASS_DICT = dict()


def register_wrapper_class(cls):
    WRAPPER_CLASS_DICT[cls.__name__] = cls
    return cls


class BaseWrapper(nn.Module):
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
    def __init__(self, classifier, pre_transform_params=None, compressor=None,
                 post_transform_params=None, analysis_config=None):
        super().__init__(analysis_config['analyzer_configs'])
        self.analyzes_after_pre_transform = analysis_config.get('analyzes_after_pre_transform', False)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.pre_transform = build_transform(pre_transform_params)
        self.compressor = compressor
        self.classifier = classifier
        self.post_transform = build_transform(post_transform_params)

    def forward(self, x):
        if self.pre_transform is not None:
            x = self.pre_transform(x)
            if self.analyzes_after_pre_transform:
                self.analyze(x)

        if self.compressor is not None:
            compressed_obj = self.compressor.compress(x)
            if not self.training and self.analyzes_after_compress:
                self.analyze(compressed_obj)
            x = self.compressor.decompress(compressed_obj)

        if self.post_transform is not None:
            x = self.post_transform(x)
        return self.classifier(x)
