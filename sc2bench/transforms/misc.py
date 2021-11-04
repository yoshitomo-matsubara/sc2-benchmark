from torch import nn
from torchdistill.datasets.transform import register_transform_class

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
