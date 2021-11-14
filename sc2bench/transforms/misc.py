import collections

import torch
from PIL.Image import Image
from torch import nn
from torch._six import string_classes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torchdistill.datasets.collator import register_collate_func
from torchdistill.datasets.transform import register_transform_class
from torchvision.transforms.functional import pad

MISC_TRANSFORM_MODULE_DICT = dict()


def register_misc_transform_module(cls):
    """
    Args:
        cls (class): codec transform module to be registered.

    Returns:
        cls (class): registered codec transform module.
    """
    MISC_TRANSFORM_MODULE_DICT[cls.__name__] = cls
    register_transform_class(cls)
    return cls


@register_collate_func
def default_collate_w_pillow(batch):
    r"""Puts each data field into a tensor or PIL Image with outer dimension batch size"""
    # Extended `default_collate` function in PyTorch

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate_w_pillow([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate_w_pillow([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_w_pillow(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_w_pillow(samples) for samples in transposed]
    elif isinstance(elem, Image):
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


@register_misc_transform_module
class ClearTargetTransform(nn.Module):
    """
    Transform module that replaces target with an empty list.
    """
    def __init__(self):
        super().__init__()

    def forward(self, sample, *args):
        """
        Args:
            sample (PIL Image or Tensor): input sample.

        Returns:
            tuple: a pair of transformed sample and original target.
        """
        return sample, list()


@register_misc_transform_module
class AdaptivePad(nn.Module):
    """
    Transform module that adaptively determines the size of padded sample.
    Args:
        fill (int): padded value.
        padding_mode (str): padding mode passed to pad module.
        factor (int): factor value for the padded input sample.
    """
    def __init__(self, fill=0, padding_mode='constant', factor=128):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode
        self.factor = factor

    def forward(self, x):
        """
        Args:
            x (PIL Image or Tensor): input sample.

        Returns:
            PIL Image or Tensor: padded input sample.
        """

        height, width = x.shape[-2:]
        vertical_pad_size = 0 if height % self.factor == 0 else int((height // self.factor + 1) * self.factor - height)
        horizontal_pad_size = 0 if width % self.factor == 0 else int((width // self.factor + 1) * self.factor - width)
        padded_vertical_size = vertical_pad_size + height
        padded_horizontal_size = horizontal_pad_size + width
        assert padded_vertical_size % self.factor == 0 and padded_horizontal_size % self.factor == 0, \
            'padded vertical and horizontal sizes ({}, {}) should be ' \
            'factor of {}'.format(padded_vertical_size, padded_horizontal_size, self.factor)
        padding = [horizontal_pad_size // 2, vertical_pad_size // 2]
        return pad(x, padding, self.fill, self.padding_mode)
