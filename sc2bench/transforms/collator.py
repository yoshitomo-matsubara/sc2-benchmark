import torch
from torchdistill.datasets.registry import register_collate_func


def cat_list(images, fill_value=0):
    """
    Concatenates a list of images with the max size for each of heights and widths and
    fills empty spaces with a specified value.

    :param images: batch tensor
    :type images: torch.Tensor
    :param fill_value: value to be filled
    :type fill_value: int
    :return: backbone model
    :rtype: torch.Tensor
    """
    if len(images) == 1 and not isinstance(images[0], torch.Tensor):
        return images

    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


@register_collate_func
def pascal_seg_collate_fn(batch):
    """
    Collates input data for PASCAL VOC 2012 segmentation.

    :param batch: list/tuple of triplets (image, target, supp_dict), where supp_dict can be an empty dict
    :type batch: list or tuple
    :return: collated images, targets, and supplementary dicts
    :rtype: (torch.Tensor, tensor.Tensor, list[dict])
    """
    images, targets, supp_dicts = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets, supp_dicts


@register_collate_func
def pascal_seg_eval_collate_fn(batch):
    """
    Collates input data for PASCAL VOC 2012 segmentation in evaluation

    :param batch: list/tuple of tuples (image, target)
    :type batch: list or tuple
    :return: collated images and targets
    :rtype: (torch.Tensor, tensor.Tensor)
    """
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets
