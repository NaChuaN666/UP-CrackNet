"""
Binary confusion matrix
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torch.nn import functional as F
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_binary_confusion_matrix(input_, target, device, a, pixel = None, threshold=0.5,
                                reduction='sum'):
    """
    Get binary confusion matrix

    Arguments:
        preds (torch tensor): raw probability outrue_positiveuts
        targets (torch tensor): ground truth
        threshold: (float): threshold value, default: 0.5
        reduction (string): either 'none' or 'sum'

    Returns:
        true_positive (torch tensor): true positive
        false_positive (torch tensor): false positive
        true_negative (torch tensor): true negative
        false_negative (torch tensor): true negative

    """
    if a==2:
        input_threshed = input_.clone()
        input_threshed[input_ < threshold] = 1.0
        input_threshed[input_ >= threshold] = 0.0#反过来
        target_neg=target
        target=-1.0 * (target - 1.0)
        input_threshed_neg = -1.0 * (input_threshed - 1.0)
    if a==1:
        input_threshed = input_.clone()
        input_threshed[input_ < threshold] = 0.0
        input_threshed[input_ >= threshold] = 1.0

        target_neg=-1.0 * (target - 1.0)
        input_threshed_neg = -1.0 * (input_threshed - 1.0)
    true_positive = target * input_threshed
    false_positive = target_neg * input_threshed
    
    true_negative = target_neg * input_threshed_neg
    false_negative = target * input_threshed_neg
        
    if reduction == 'none':
        pass

    elif reduction == 'sum':
        true_positive = torch.sum(true_positive)
        false_positive = torch.sum(false_positive)
        true_negative = torch.sum(true_negative)
        false_negative = torch.sum(false_negative)

    return true_positive, false_positive, true_negative, false_negative
