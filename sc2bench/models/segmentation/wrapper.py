import torch
from torchdistill.common.main_util import load_ckpt
from torchdistill.datasets.util import build_transform
from torchvision.transforms.functional import crop

from .registry import load_segmentation_model
from ..registry import get_compression_model
from ..wrapper import register_wrapper_class, WRAPPER_CLASS_DICT
from ...analysis import AnalyzableModule


@register_wrapper_class
class CodecInputCompressionSegmentationModel(AnalyzableModule):
    """
    A wrapper module for codec input compression model followed by a segmentation model.

    :param segmentation_model: semantic segmentation model
    :type segmentation_model: nn.Module
    :param device: torch device
    :type device: torch.device or str
    :param codec_params: transform sequence configuration for codec
    :type codec_params: dict or None
    :param post_transform_params: post-transform parameters
    :type post_transform_params: dict or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, segmentation_model, device, codec_params=None,
                 post_transform_params=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.codec_encoder_decoder = build_transform(codec_params)
        self.device = device
        self.segmentation_model = segmentation_model
        self.post_transform = build_transform(post_transform_params)

    def forward(self, x):
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
        return self.segmentation_model(x)


@register_wrapper_class
class NeuralInputCompressionSegmentationModel(AnalyzableModule):
    """
    A wrapper module for neural input compression model followed by a segmentation model.

    :param segmentation_model: semantic segmentation model
    :type segmentation_model: nn.Module
    :param pre_transform_params: pre-transform parameters
    :type pre_transform_params: dict or None
    :param compression_model: compression model
    :type compression_model: nn.Module or None
    :param uses_cpu4compression_model: whether to use CPU instead of GPU for `comoression_model`
    :type uses_cpu4compression_model: bool
    :param post_transform_params: post-transform parameters
    :type post_transform_params: dict or None
    :param analysis_config: analysis configuration
    :type analysis_config: dict or None
    """
    def __init__(self, segmentation_model, pre_transform_params=None, compression_model=None,
                 uses_cpu4compression_model=False, post_transform_params=None, analysis_config=None, **kwargs):
        if analysis_config is None:
            analysis_config = dict()

        super().__init__(analysis_config.get('analyzer_configs', list()))
        self.analyzes_after_pre_transform = analysis_config.get('analyzes_after_pre_transform', False)
        self.analyzes_after_compress = analysis_config.get('analyzes_after_compress', False)
        self.pre_transform = build_transform(pre_transform_params)
        self.compression_model = compression_model
        self.uses_cpu4compression_model = uses_cpu4compression_model
        self.segmentation_model = segmentation_model
        self.post_transform = build_transform(post_transform_params)

    def use_cpu4compression(self):
        """
        Changes the device of the compression model to CPU.
        """
        if self.uses_cpu4compression_model and self.compression_model is not None:
            self.compression_model = self.compression_model.cpu()

    def forward(self, x):
        org_patch_size = None
        if self.pre_transform is not None:
            x = self.pre_transform(x)
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], tuple):
                org_patch_size = x[1]
                x = x[0]

            if not self.training and self.analyzes_after_pre_transform:
                self.analyze(x)

        if self.compression_model is not None:
            compressed_obj = self.compression_model.compress(x)
            if not self.training and self.analyzes_after_compress:
                compressed_data = compressed_obj if org_patch_size is None else (compressed_obj, org_patch_size)
                self.analyze(compressed_data)
            x = self.compression_model.decompress(**compressed_obj)
            if isinstance(x, dict):
                x = x['x_hat']

        if self.post_transform is not None:
            if org_patch_size is not None:
                x = crop(x, 0, 0, org_patch_size[0], org_patch_size[1])
            x = self.post_transform(x)
        return self.segmentation_model(x)


def get_wrapped_segmentation_model(wrapper_model_config, device):
    """
    Gets a wrapped semantic segmentation model

    :param wrapper_model_config: wrapper model configuration
    :type wrapper_model_config: dict
    :param device: torch device
    :type device: torch.device
    :return: model: wrapped semantic segmentation model
    :rtype: model: nn.Module
    """
    wrapper_model_name = wrapper_model_config['name']
    if wrapper_model_name not in WRAPPER_CLASS_DICT:
        raise ValueError('wrapper_model_name `{}` is not expected'.format(wrapper_model_name))

    compression_model_config = wrapper_model_config.get('compression_model', None)
    compression_model = get_compression_model(compression_model_config, device)
    segmentation_model_config = wrapper_model_config['segmentation_model']
    model = load_segmentation_model(segmentation_model_config, device)
    wrapped_model = WRAPPER_CLASS_DICT[wrapper_model_name](model, compression_model=compression_model, device=device,
                                                           **wrapper_model_config['params'])
    ckpt_file_path = wrapper_model_config.get('ckpt', None)
    if ckpt_file_path is not None:
        load_ckpt(ckpt_file_path, model=wrapped_model, strict=False)
    return wrapped_model
