from collections import OrderedDict

from torch.nn import functional
from torchdistill.common.constant import def_logger
from ..backbone import check_if_updatable
from ...analysis import AnalyzableModule, check_if_analyzable

logger = def_logger.getChild(__name__)


class UpdatableSegmentationModel(AnalyzableModule):
    """
    An abstract class for updatable semantic segmentation model.

    :param analyzer_configs: list of analysis configurations
    :type analyzer_configs: list[dict]
    """
    def __init__(self, analyzer_configs=None):
        super().__init__(analyzer_configs)
        self.bottleneck_updated = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, **kwargs):
        """
        Updates compression-specific parameters like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.update>`_.

        This should be overridden by all subclasses.
        """
        raise NotImplementedError()

    def get_aux_module(self, **kwargs):
        """
        Returns an auxiliary module to compute auxiliary loss if necessary like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.aux_loss>`_.

        This should be overridden by all subclasses.
        """
        raise NotImplementedError()


class BaseSegmentationModel(UpdatableSegmentationModel):
    # Referred to the base implementation at https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/_utils.py
    __constants__ = ['aux_classifier']
    """
    A base, updatable segmentation model.

    :param backbone: backbone model (usually a classification model)
    :type backbone: nn.Module
    :param classifier: classification model
    :type classifier: nn.Module
    :param aux_classifier: auxiliary classification model to be used during training only
    :type aux_classifier: nn.Module or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
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
        """
        Updates compression-specific parameters like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.update>`_.
        Needs to be called once after training to be able to later perform the evaluation with an actual entropy coder.
        """
        if not check_if_updatable(self.backbone):
            raise KeyError(f'`backbone` {type(self)} is not updatable')
        self.backbone.update()

    def get_aux_module(self, **kwargs):
        """
        Returns an auxiliary module to compute auxiliary loss if necessary like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.aux_loss>`_.

        :return: auxiliary module
        :rtype: nn.Module
        """
        return self.backbone.get_aux_module()

    def activate_analysis(self):
        """
        Activates the analysis mode.

        Should be called after training model.
        """
        self.activated_analysis = True
        if check_if_analyzable(self.backbone):
            self.backbone.activate_analysis()

    def deactivate_analysis(self):
        """
        Deactivates the analysis mode.
        """
        self.activated_analysis = False
        self.backbone.deactivate_analysis()
        if check_if_analyzable(self.backbone):
            self.backbone.deactivate_analysis()

    def analyze(self, compressed_obj):
        """
        Analyzes a given compressed object (e.g., file size of the compressed object).

        :param compressed_obj: compressed object to be analyzed
        :type compressed_obj: Any
        """
        if not self.activated_analysis:
            return

        for analyzer in self.analyzers:
            analyzer.analyze(compressed_obj)
        if check_if_analyzable(self.backbone):
            self.backbone.analyze(compressed_obj)

    def summarize(self):
        """
        Summarizes the results that the configured analyzers store.
        """
        for analyzer in self.analyzers:
            analyzer.summarize()
        if check_if_analyzable(self.backbone):
            self.backbone.summarize()

    def clear_analysis(self):
        """
        Clears the results that the configured analyzers store.
        """
        for analyzer in self.analyzers:
            analyzer.clear()
        if check_if_analyzable(self.backbone):
            self.backbone.clear_analysis()


def check_if_updatable_segmentation_model(model):
    """
    Checks if the given semantic segmentation model is updatable.

    :param model: semantic segmentation model
    :type model: nn.Module
    :return: True if the model is updatable, False otherwise
    :rtype: bool
    """
    return isinstance(model, UpdatableSegmentationModel)
