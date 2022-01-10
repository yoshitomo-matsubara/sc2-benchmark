from typing import Dict, Optional, List

from torch import nn, Tensor
from torchdistill.common.constant import def_logger
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock

from ..backbone import FeatureExtractionBackbone
from ...analysis import AnalyzableModule

logger = def_logger.getChild(__name__)


class UpdatableDetectionModel(AnalyzableModule):
    def __init__(self, analyzer_configs=None):
        super().__init__(analyzer_configs)
        self.bottleneck_updated = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, **kwargs):
        raise NotImplementedError()

    def get_aux_module(self, **kwargs):
        raise NotImplementedError()


class UpdatableBackboneWithFPN(UpdatableDetectionModel):
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
        if self.analyzable_layer_key is None or self.analyzable_layer_key not in self._modules:
            return False
        return True

    def update(self):
        self.body.update()
        self.bottleneck_updated = True

    def get_aux_module(self):
        return self.body.get_aux_module()


def check_if_updatable_detection_model(model):
    return isinstance(model, UpdatableDetectionModel)
