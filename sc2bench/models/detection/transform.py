from typing import List, Tuple, Dict, Optional

from torch import Tensor
from torchdistill.datasets.util import build_transform
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms.functional import to_pil_image, to_tensor, crop

from ...analysis import AnalyzableModule
from ...transforms.misc import AdaptivePad


class RCNNTransformWithCompression(GeneralizedRCNNTransform, AnalyzableModule):
    """
    An R-CNN Transform with codec-based or model-based compression

    :param transform: performs the data transformation from the inputs to feed into the model
    :type transform: nn.Module
    :param device: torch device
    :type device: torch.device or str
    :param codec_params: codec parameters
    :type codec_params: dict
    :param analyzer_configs: a list of analysis configurations
    :type analyzer_configs: list[dict]
    :param analyzes_after_compress: run analysis with `analyzer_configs` if `True`
    :type analyzes_after_compress: bool
    :param compression_model: a compression model
    :type compression_model: nn.Module or None
    :param uses_cpu4compression_model: whether to use CPU instead of GPU for `comoression_model`
    :type uses_cpu4compression_model: bool
    :param pre_transform_params: pre-transform parameters
    :type pre_transform_params: dict or None
    :param post_transform_params: post-transform parameters
    :type post_transform_params: dict or None
    :param adaptive_pad_kwargs: keyword arguments for AdaptivePad
    :type adaptive_pad_kwargs: dict or None
    """
    # Referred to https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py
    def __init__(self, transform, device, codec_params, analyzer_configs, analyzes_after_compress=False,
                 compression_model=None, uses_cpu4compression_model=False, pre_transform_params=None,
                 post_transform_params=None, adaptive_pad_kwargs=None):
        GeneralizedRCNNTransform.__init__(self, transform.min_size, transform.max_size,
                                          transform.image_mean, transform.image_std)
        AnalyzableModule.__init__(self, analyzer_configs)
        self.device = device
        self.codec_encoder_decoder = build_transform(codec_params)
        self.analyzes_after_compress = analyzes_after_compress
        self.pre_transform = build_transform(pre_transform_params)
        self.post_transform = build_transform(post_transform_params)
        if uses_cpu4compression_model:
            compression_model = compression_model.cpu()

        self.compression_model = compression_model
        self.uses_cpu4compression_model = uses_cpu4compression_model
        self.adaptive_pad = AdaptivePad(**adaptive_pad_kwargs) if isinstance(adaptive_pad_kwargs, dict) else None

    def compress_by_codec(self, org_img):
        """
        Convert a tensor to an image and compress-decompress it by codec.

        :param org_img: an image tensor
        :type org_img: torch.Tensor
        :return: a compressed-and-decompressed image tensor
        :rtype: torch.Tensor
        """
        pil_img = to_pil_image(org_img, mode='RGB')
        pil_img, file_size = self.codec_encoder_decoder(pil_img)
        if not self.training:
            self.analyze(file_size)
        return to_tensor(pil_img).to(org_img.device)

    def compress_by_model(self, org_img):
        """
        Convert a tensor to an image and compress-decompress it by model.

        :param org_img: an image tensor
        :type org_img: torch.Tensor
        :return: a compressed-and-decompressed image tensor
        :rtype: torch.Tensor
        """
        org_img = org_img.unsqueeze(0)
        org_height, org_width = None, None
        if self.adaptive_pad is not None:
            org_height, org_width = org_img.shape[-2:]
            org_img = self.adaptive_pad(org_img)

        compressed_obj = self.compression_model.compress(org_img)
        if not self.training and self.analyzes_after_compress:
            compressed_data = compressed_obj if org_height is None or org_width is None \
                else (compressed_obj, org_height, org_width)
            self.analyze(compressed_data)

        decompressed_obj = self.compression_model.decompress(**compressed_obj)
        decompressed_obj = decompressed_obj['x_hat']
        if org_height is not None and org_width is not None:
            decompressed_obj = crop(decompressed_obj, 0, 0, org_height, org_width)
        return decompressed_obj.squeeze(0)

    def compress(self, org_img):
        """
        Apply `pre_transform` to an image tensor, compress and decompress it, and apply `post_transform` to
        the compressed-decompressed image tensor.

        :param org_img: an image tensor
        :type org_img: torch.Tensor
        :return: a compressed-and-decompressed image tensor
        :rtype: torch.Tensor
        """
        if self.pre_transform is not None:
            org_img = self.pre_transform(org_img)

        org_device = org_img.device
        if self.uses_cpu4compression_model:
            org_img = org_img.cpu()

        org_img = self.compress_by_codec(org_img) if self.compression_model is None else self.compress_by_model(org_img)
        if self.uses_cpu4compression_model:
            org_img = org_img.to(org_device)

        if self.post_transform is not None:
            org_img = self.post_transform(org_img)
        return org_img

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image, target_index = self.resize(image, target_index)
            shape_before_compression = image.shape
            image = self.compress(image)
            shape_after_compression = image.shape
            assert shape_after_compression == shape_before_compression, \
                'Compression should not change tensor shape {} -> {}'.format(shape_before_compression,
                                                                             shape_after_compression)
            image = self.normalize(image)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets
