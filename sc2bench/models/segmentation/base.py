from collections import OrderedDict

from torch.nn import functional
from torchdistill.common.constant import def_logger
from ..backbone import check_if_updatable
from ...analysis import AnalyzableModule, check_if_analyzable

logger = def_logger.getChild(__name__)


class UpdatableSegmentationModel(AnalyzableModule):
    def __init__(self, analyzer_configs=None):
        super().__init__(analyzer_configs)
        self.bottleneck_updated = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, **kwargs):
        raise NotImplementedError()

    def get_aux_module(self, **kwargs):
        raise NotImplementedError()


def check_if_updatable_segmentation_model(model):
    return isinstance(model, UpdatableSegmentationModel)


class BaseSegmentationModel(UpdatableSegmentationModel):
    # Referred to the base implementation at https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/_utils.py
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None, analysis_config=None):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features['out']
        x = self.classifier(x)
        x = functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result['out'] = x

        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            x = functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result

    def update(self, **kwargs):
        if not check_if_updatable(self.backbone):
            raise KeyError(f'`backbone` {type(self)} is not updatable')
        self.backbone.update()

    def get_aux_module(self, **kwargs):
        return self.backbone.get_aux_module()

    def activate_analysis(self):
        self.activated_analysis = True
        if check_if_analyzable(self.backbone):
            self.backbone.activate_analysis()

    def deactivate_analysis(self):
        self.activated_analysis = False
        self.backbone.deactivate_analysis()
        if check_if_analyzable(self.backbone):
            self.backbone.deactivate_analysis()

    def analyze(self, compressed_obj):
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj)
        if check_if_analyzable(self.backbone):
            self.backbone.analyze(compressed_obj)

    def summarize(self):
        for analyzer in self.analyzers:
            analyzer.summarize()
        if check_if_analyzable(self.backbone):
            self.backbone.summarize()

    def clear_analysis(self):
        for analyzer in self.analyzers:
            analyzer.clear()
        if check_if_analyzable(self.backbone):
            self.backbone.clear_analysis()
