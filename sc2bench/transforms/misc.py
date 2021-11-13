from torch import nn
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
