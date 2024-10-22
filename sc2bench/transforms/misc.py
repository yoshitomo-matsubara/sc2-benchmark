import collections

import numpy as np
import torch
from PIL.Image import Image
from torch import nn
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchdistill.common import tensor_util
from torchdistill.datasets.registry import register_collate_func, register_transform
from torchvision.transforms import functional as F
from torchvision.transforms.functional import pad

MISC_TRANSFORM_MODULE_DICT = dict()


def register_misc_transform_module(cls):
    """
    Registers a miscellaneous transform class.

    :param cls: miscellaneous transform class to be registered
    :type cls: class
    :return: registered miscellaneous transform class
    :rtype: class
    """
    MISC_TRANSFORM_MODULE_DICT[cls.__name__] = cls
    register_transform(cls)
    return cls


@register_collate_func
def default_collate_w_pil(batch):
    """
    Puts each data field into a tensor or PIL Image with outer dimension batch size.

    :param batch: single batch to be collated
    :return: collated batch
    """
    # Extended `default_collate` function in PyTorch

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate_w_pil([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate_w_pil([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_w_pil(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_w_pil(samples) for samples in transposed]
    elif isinstance(elem, Image):
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


@register_misc_transform_module
class ClearTargetTransform(nn.Module):
    """
    A transform module that replaces target with an empty list.
    """
    def __init__(self):
        super().__init__()

    def forward(self, sample, *args):
        """
        Replaces target data field with an empty list.

        :param sample: image or image tensor
        :type sample: PIL.Image.Image or torch.Tensor
        :return: sample and an empty list
        :rtype: (PIL.Image.Image or torch.Tensor, list)
        """
        return sample, list()


@register_misc_transform_module
class AdaptivePad(nn.Module):
    """
    A transform module that adaptively determines the size of padded sample.

    :param fill: padded value
    :type fill: int
    :param padding_position: 'hw' (default) to pad left and right for padding horizontal size // 2 and top and
            bottom for padding vertical size // 2; 'right_bottom' to pad bottom and right only
    :type padding_position: str
    :param padding_mode: padding mode passed to pad module
    :type padding_mode: str
    :param factor: factor value for the padded input sample
    :type factor: int
    :param returns_org_patch_size: if True, returns the patch size of the original input
    :type returns_org_patch_size: bool
    """
    def __init__(self, fill=0, padding_position='hw', padding_mode='constant',
                 factor=128, returns_org_patch_size=False):
        super().__init__()
        self.fill = fill
        self.padding_position = padding_position
        self.padding_mode = padding_mode
        self.factor = factor
        self.returns_org_patch_size = returns_org_patch_size

    def forward(self, x):
        """
        Adaptively determines the size of padded image or image tensor.

        :param x: image or image tensor
        :type x: PIL.Image.Image or torch.Tensor
        :return: padded image or image tensor, and the patch size of the input (height, width)
                    if returns_org_patch_size=True
        :rtype: PIL.Image.Image or torch.Tensor or (PIL.Image.Image or torch.Tensor, list[int, int])
        """
        height, width = x.shape[-2:]
        vertical_pad_size = 0 if height % self.factor == 0 else int((height // self.factor + 1) * self.factor - height)
        horizontal_pad_size = 0 if width % self.factor == 0 else int((width // self.factor + 1) * self.factor - width)
        padded_vertical_size = vertical_pad_size + height
        padded_horizontal_size = horizontal_pad_size + width
        assert padded_vertical_size % self.factor == 0 and padded_horizontal_size % self.factor == 0, \
            'padded vertical and horizontal sizes ({}, {}) should be ' \
            'factor of {}'.format(padded_vertical_size, padded_horizontal_size, self.factor)
        padding = [horizontal_pad_size // 2, vertical_pad_size // 2] if self.padding_position == 'equal_side' \
            else [0, 0, horizontal_pad_size, vertical_pad_size]
        x = pad(x, padding, self.fill, self.padding_mode)
        if self.returns_org_patch_size:
            return x, (height, width)
        return x


@register_misc_transform_module
class CustomToTensor(nn.Module):
    """
    A customized ToTensor module that can be applied to sample and target selectively.

    :param converts_sample: if True, applies to_tensor to sample
    :type converts_sample: bool
    :param converts_target: if True, applies torch.as_tensor to target
    :type converts_target: bool
    """
    def __init__(self, converts_sample=True, converts_target=True):
        super().__init__()
        self.converts_sample = converts_sample
        self.converts_target = converts_target

    def __call__(self, image, target):
        if self.converts_sample:
            image = F.to_tensor(image)

        if self.converts_target:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


@register_misc_transform_module
class SimpleQuantizer(nn.Module):
    """
    A module to quantize tensor with its half() function if num_bits=16 (FP16) or
    Jacob et al.'s method if num_bits=8 (INT8 + one FP32 scale parameter).

    Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko: `"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" <https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html>`_ @ CVPR 2018 (2018)

    :param num_bits: number of bits for quantization
    :type num_bits: int
    """
    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, z):
        """
        Quantizes tensor.

        :param z: tensor
        :type z: torch.Tensor
        :return: quantized tensor
        :rtype: torch.Tensor or torchdistill.common.tensor_util.QuantizedTensor
        """
        return z.half() if self.num_bits == 16 else tensor_util.quantize_tensor(z, self.num_bits)


@register_misc_transform_module
class SimpleDequantizer(nn.Module):
    """
    A module to dequantize quantized tensor in FP32. If num_bits=8, it uses Jacob et al.'s method.

    Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko: `"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" <https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html>`_ @ CVPR 2018 (2018)

    :param num_bits: number of bits used for quantization
    :type num_bits: int
    """
    def __init__(self, num_bits):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, z):
        """
        Dequantizes quantized tensor.

        :param z: quantized tensor
        :type z: torch.Tensor or torchdistill.common.tensor_util.QuantizedTensor
        :return: dequantized tensor
        :rtype: torch.Tensor
        """
        return z.float() if self.num_bits == 16 else tensor_util.dequantize_tensor(z)
