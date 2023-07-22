from typing import Dict, Optional, List

from torch import nn, Tensor
from torchdistill.common.constant import def_logger
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock

from ..backbone import FeatureExtractionBackbone
from ...analysis import AnalyzableModule

logger = def_logger.getChild(__name__)


class UpdatableDetectionModel(AnalyzableModule):
    """
    An abstract class for updatable object detection model.

    :param analyzer_configs: a list of analysis configurations
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


class UpdatableBackboneWithFPN(UpdatableDetectionModel):
    """
    An updatable backbone model with feature pyramid network (FPN).

    :param backbone: a backbone model (usually a classification model)
    :type backbone: nn.Module
    :param return_layer_dict: a mapping from name of module to return its output to a specified key
    :type return_layer_dict: dict
    :param in_channels_list: number of channels for each feature map that is passed to the module for FPN
    :type in_channels_list: list[int]
    :param out_channels: number of channels of the FPN representation
    :type out_channels: int
    :param extra_blocks: if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    :type extra_blocks: ExtraFPNBlock or None
    :param analyzer_configs: a list of analysis configurations
    :type analyzer_configs: list[dict]
    :param analyzes_after_compress: run analysis with `analyzer_configs` if True
    :type analyzes_after_compress: bool
    :param analyzable_layer_key: key of analyzable layer
    :type analyzable_layer_key: str
    """
    # Referred to https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
    def __init__(
        self,
        backbone: nn.Module,
        return_layer_dict: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        analyzer_configs: List[Dict] = None,
        analyzes_after_compress: bool = False,
        analyzable_layer_key: str = None
    ) -> None:
        super().__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if analyzer_configs is None:
            analyzer_configs = list()

        self.body = FeatureExtractionBackbone(backbone, return_layer_dict=return_layer_dict,
                                              analyzer_configs=analyzer_configs,
                                              analyzes_after_compress=analyzes_after_compress,
                                              analyzable_layer_key=analyzable_layer_key)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x

    def check_if_updatable(self):
        """
        Checks if this module is updatable with respect to CompressAI modules.

        :return: True if the model is updatable, False otherwise
        :rtype: bool
        """
        if self.analyzable_layer_key is None or self.analyzable_layer_key not in self._modules:
            return False
        return True

    def update(self):
        """
        Updates compression-specific parameters like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.update>`_.
        Needs to be called once after training to be able to later perform the evaluation with an actual entropy coder.
        """
        self.body.update()
        self.bottleneck_updated = True

    def get_aux_module(self):
        """
        Returns an auxiliary module to compute auxiliary loss if necessary like `CompressAI models do <https://interdigitalinc.github.io/CompressAI/models.html#compressai.models.CompressionModel.aux_loss>`_.

        :return: an auxiliary module
        :rtype: nn.Module
        """
        return self.body.get_aux_module()


def check_if_updatable_detection_model(model):
    """
    Checks if the given object detection model is updatable.

    :param model: an object detection model
    :type model: nn.Module
    :return: True if the model is updatable, False otherwise
    :rtype: bool
    """
    return isinstance(model, UpdatableDetectionModel)
