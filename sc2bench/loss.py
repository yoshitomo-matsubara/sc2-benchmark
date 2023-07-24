from torch import nn
from torchdistill.losses.single import register_single_loss
from torchdistill.losses.util import register_func2extract_org_output


@register_func2extract_org_output
def extract_org_loss_dict(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    """
    Extracts loss(es) from student_outputs inside `TrainingBox` or `DistillationBox` in `torchdistill`.

    :param org_criterion: not used
    :type org_criterion: nn.Module
    :param student_outputs: student models' output
    :type student_outputs: dict or Any
    :param teacher_outputs: not used
    :type teacher_outputs: Any
    :param targets: not used
    :type targets: Any
    :param uses_teacher_output: not used
    :type uses_teacher_output: bool
    :return: original loss dict
    :rtype: class
    """
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        org_loss_dict.update(student_outputs)
    return org_loss_dict


@register_func2extract_org_output
def extract_org_segment_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    """
    Computes loss(es) using the original loss module inside `TrainingBox` or `DistillationBox` in `torchdistill`
    for semantic segmentation models in `torchvision`.

    :param org_criterion: original loss module
    :type org_criterion: nn.Module
    :param student_outputs: student models' output
    :type student_outputs: dict or Any
    :param teacher_outputs: not used
    :type teacher_outputs: Any
    :param targets: targets
    :type targets: Any
    :param uses_teacher_output: not used
    :type uses_teacher_output: bool
    :return: original loss dict
    :rtype: class
    """
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        sub_loss_dict = dict()
        for key, outputs in student_outputs.items():
            sub_loss_dict[key] = org_criterion(outputs, targets)

        org_loss = sub_loss_dict['out']
        if 'aux' in sub_loss_dict:
            org_loss += 0.5 * sub_loss_dict['aux']
        org_loss_dict['total'] = org_loss
    return org_loss_dict


@register_single_loss
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
