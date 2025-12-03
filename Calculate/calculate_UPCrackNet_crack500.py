import os
import numpy as np
import torch
from PIL import Image
from fvcore.nn import FlopCountAnalysis
from calculate_metrics import calculate_metrics, metrics
import sys
sys.path.append('check')
from average_meter import AverageMeter
import logging

def load_image_as_tensor(image_path):
    """
    Load an image and convert it to a binary tensor.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)
    image = torch.from_numpy(image).float()

    # Convert to binary (0, 1)
    image = torch.where(image > 127, 1.0, 0.0)
    
    return image

def calculate2_metrics(pred_folder, gt_folder):
    """
    Iterate over the prediction and groundtruth folders, calculate metrics for each image pair.
    """
    epoch_acc = AverageMeter()
    epoch_recall = AverageMeter()
    epoch_precision = AverageMeter()
    epoch_f1_score = AverageMeter()
    epoch_iou = AverageMeter()
    epoch_aiu = AverageMeter()
    TP = []
    TN = []
    FP = []
    FN = []
    TP1 = []
    TN1 = []
    FP1 = []
    FN1 = []

    for filename in os.listdir(pred_folder):
        pred_path = os.path.join(pred_folder, filename)
        gt_path = os.path.join(gt_folder, filename)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            pred = load_image_as_tensor(pred_path)
            gt = load_image_as_tensor(gt_path)
            
            curr_TP, curr_FP, curr_TN, curr_FN,curr_TP1, curr_FP1, curr_TN1, curr_FN1 = calculate_metrics(pred, gt, device)
            TP.append(curr_TP)
            TN.append(curr_TN)
            FP.append(curr_FP)
            FN.append(curr_FN) 
            TP1.append(curr_TP1)
            TN1.append(curr_TN1)
            FP1.append(curr_FP1)
            FN1.append(curr_FN1)

    # 将列表中的张量累加
    tp = torch.sum(torch.stack(TP), dim=0)
    tn = torch.sum(torch.stack(TN), dim=0)
    fp = torch.sum(torch.stack(FP), dim=0)
    fn = torch.sum(torch.stack(FN), dim=0)  
    tp1 = torch.sum(torch.stack(TP1), dim=0)
    tn1 = torch.sum(torch.stack(TN1), dim=0)
    fp1 = torch.sum(torch.stack(FP1), dim=0)
    fn1 = torch.sum(torch.stack(FN1), dim=0)  
    (curr_acc, curr_recall,curr_precision,curr_f1_score, curr_iou, curr_aiu) = metrics(tp, fp, tn, fn,tp1, fp1, tn1, fn1)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('acc: %f | recall: %f | precision: %f | f1_score: %f | IOU: %f | AIU: %f|',curr_acc, curr_recall, curr_precision, curr_f1_score, curr_iou, curr_aiu)   

def main():
    
    pred_folder = '/root/autodl-tmp/UP-CrackNet-main/outputs_all/best'
    
    gt_folder = '/root/autodl-tmp/UP-CrackNet-main/dataset/crack500_test/test_folder_all/label'

    metrics = calculate2_metrics(pred_folder, gt_folder)


if __name__ == "__main__":
    main()
