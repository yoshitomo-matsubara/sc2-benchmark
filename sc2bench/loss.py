from torch import nn
from torchdistill.losses.single import register_single_loss
from torchdistill.losses.util import register_func2extract_org_output


@register_func2extract_org_output
def extract_org_loss_dict(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
    org_loss_dict = dict()
    if isinstance(student_outputs, dict):
        org_loss_dict.update(student_outputs)
    return org_loss_dict


@register_func2extract_org_output
def extract_org_segment_loss(org_criterion, student_outputs, teacher_outputs, targets, uses_teacher_output, **kwargs):
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
class BppBasedLoss(nn.Module):
    def __init__(self, compressor_module_path, beta, reduction='mean'):
        super().__init__()
        self.compressor_module_path = compressor_module_path
        self.beta = beta
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, student_io_dict, *args, **kwargs):
        compressor_module_dict = student_io_dict[self.compressor_module_path]
        compressor_inputs = compressor_module_dict['input']
        reconstructed_inputs, likelihoods = compressor_module_dict['output']
        reconstruction_loss = self.mse(reconstructed_inputs, compressor_inputs)
        n, _, h, w = compressor_inputs.shape
        num_pixels = n * h * w
        if self.reduction == 'sum':
            bpp = -likelihoods.log2().sum()
        elif self.reduction == 'batchmean':
            bpp = -likelihoods.log2().sum() / n
        else:
            bpp = -likelihoods.log2().sum() / num_pixels
        return reconstruction_loss + self.beta * bpp


@register_single_loss
class BppLoss(nn.Module):
    def __init__(self, entropy_module_path, reduction='mean'):
        super().__init__()
        self.entropy_module_path = entropy_module_path
        self.reduction = reduction

    def forward(self, student_io_dict, *args, **kwargs):
        entropy_module_dict = student_io_dict[self.entropy_module_path]
        reconstructed_inputs, likelihoods = entropy_module_dict['output']
        n, _, h, w = reconstructed_inputs.shape
        num_pixels = n * h * w
        if self.reduction == 'sum':
            bpp = -likelihoods.log2().sum()
        elif self.reduction == 'batchmean':
            bpp = -likelihoods.log2().sum() / n
        else:
            bpp = -likelihoods.log2().sum() / num_pixels
        return bpp
