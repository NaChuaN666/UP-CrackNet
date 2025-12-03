"""
Statistical metrics based on binary confusion matrix
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from utils import tensor2float
import torch 

def tensor2float(input_):
    """
    Convert a one element tensor into float
    """
    if isinstance(input_, torch.Tensor):
        if input_.numel() == 1:
            output = input_.item()
        else:
            raise ValueError

    return output

def get_accuracy(true_positive, false_positive, true_negative, false_negative,
                 epsilon=1e-8):
    """
    Accuracy
    """
    true_positive, false_positive, true_negative, false_negative = list(
        map(tensor2float, [true_positive, false_positive,
                           true_negative, false_negative])
    )

    try:
        acc = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative)

    except ZeroDivisionError:
        acc = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive +
            false_negative + epsilon)

    return acc


def get_true_positive_rate(true_positive, false_negative, epsilon=1e-8):
    """
    True Positive Rate (Sensitivity, Recall)
    """
    true_positive, false_negative = list(
        map(tensor2float, [true_positive, false_negative])
    )

    try:
        tpr = true_positive / (true_positive + false_negative)

    except ZeroDivisionError:
        tpr = true_positive / (true_positive + false_negative + epsilon)

    return tpr


def get_true_negative_rate(false_positive, true_negative, epsilon=1e-8):
    """
    True Negative Rate (Specificity)
    """
    false_positive, true_negative = list(
        map(tensor2float, [false_positive, true_negative])
    )

    try:
        tnr = true_negative / (true_negative + false_positive)

    except ZeroDivisionError:
        tnr = true_negative / (true_negative + false_positive + epsilon)

    return tnr


def get_precision(true_positive, false_positive, epsilon=1e-8):
    """
    Precision
    """
    true_positive, false_positive = list(
        map(tensor2float, [true_positive, false_positive])
    )

    try:
        prc = true_positive / (true_positive + false_positive)

    except ZeroDivisionError:
        prc = true_positive / (true_positive + false_positive + epsilon)

    return prc


def get_f_score(true_positive, false_positive, false_negative,
                beta=1, epsilon=1e-8):
    """
    General F score
    """

    true_positive, false_positive, false_negative = list(
        map(tensor2float, [true_positive, false_positive, false_negative])
    )

    try:
        f_beta = ((1 + beta ** 2) * true_positive) / (
            (1 + beta ** 2) * true_positive + false_positive + false_negative)

    except ZeroDivisionError:
        f_beta = ((1 + beta ** 2) * true_positive) / (
            (1 + beta ** 2) * true_positive + false_positive +
            false_negative + epsilon)

    return f_beta


def get_f1_socre(true_positive, false_positive, false_negative, epsilon=1e-8):
    """
    F1 score, harmonic mean of recall and precision
    """
    true_positive, false_positive, false_negative = list(
        map(tensor2float, [true_positive, false_positive, false_negative])
    )
    try:
        f1 = (2 * true_positive) / (2 * true_positive +
                                    false_positive + false_negative)
    except ZeroDivisionError:
        f1 = (2 * true_positive) / (2 * true_positive +
                                    false_positive + false_negative + epsilon)
    return f1


def get_iou(true_positive, false_positive, false_negative,
            epsilon=1e-8):
    """
    Intersection over union
    """
    true_positive, false_positive, false_negative = list(
        map(tensor2float, [true_positive, false_positive, false_negative])
    )

    try:
        iou = true_positive / (true_positive + false_positive + false_negative)

    except ZeroDivisionError:
        iou = true_positive / (true_positive + false_positive +
                               false_negative + epsilon)

    return iou
