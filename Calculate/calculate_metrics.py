import sys
sys.path.append('check')
from binary_confusion_matrix import get_binary_confusion_matrix
from binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, get_precision, get_f1_socre, get_iou
# from dice_coefficient import hard_dice
# from pr_curve import get_pr_curve
# from roc_curve import get_auroc, get_roc_curve
# from util.numpy_utils import tensor2numpy

def calculate_metrics(preds, targets,device):
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device =device, pixel = None, 
        threshold=0.5,
        reduction='sum',a=1)
    curr_TP1, curr_FP1, curr_TN1, curr_FN1 = get_binary_confusion_matrix(
        input_=preds, target=targets, device =device, pixel = None, 
        threshold=0.5,
        reduction='sum',a=2)
    return curr_TP, curr_FP, curr_TN, curr_FN,curr_TP1, curr_FP1, curr_TN1, curr_FN1
    
def metrics(curr_TP, curr_FP, curr_TN, curr_FN,curr_TP1, curr_FP1, curr_TN1, curr_FN1):
    curr_acc = get_accuracy(true_positive=curr_TP,
                            false_positive=curr_FP,
                            true_negative=curr_TN,
                            false_negative=curr_FN)
    curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                         false_negative=curr_FN)
    curr_precision = get_precision(true_positive=curr_TP,
                                   false_positive=curr_FP)
    curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                 false_positive=curr_FP,
                                 false_negative=curr_FN)
    curr_iou = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)
    
    curr_iou1 = get_iou(true_positive=curr_TP1,
                       false_positive=curr_FP1,
                       false_negative=curr_FN1)
    curr_aiu=(curr_iou+curr_iou1)/2
    return (curr_acc, curr_recall,curr_precision,
            curr_f1_score, curr_iou, curr_aiu)
