from torch import nn
from torchdistill.losses.mid_level import register_mid_level_loss


@register_mid_level_loss
class BppLoss(nn.Module):
    """
    Bit-per-pixel (or rate) loss.

    :param entropy_module_path: entropy module path to extract its output from io_dict
    :type entropy_module_path: str
    :param reduction: reduction type ('sum', 'batchmean', or 'mean')
    :type reduction: str or None
    """
    def __init__(self, entropy_module_path, reduction='mean'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction

    def forward(self, student_io_dict, *args, **kwargs):
        """
        Computes a rate loss.

        :param student_io_dict: io_dict of model to be trained
        :type student_io_dict: dict
        """
        entropy_module_dict = student_io_dict[self.entropy_module_path]
        intermediate_features, likelihoods = entropy_module_dict['output']
        n, _, h, w = intermediate_features.shape
        num_pixels = n * h * w
        if self.reduction == 'sum':
            bpp = -likelihoods.log2().sum()
        elif self.reduction == 'batchmean':
            bpp = -likelihoods.log2().sum() / n
        else:
            bpp = -likelihoods.log2().sum() / num_pixels
        return bpp
