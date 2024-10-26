import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
#from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
from torchmetrics import Accuracy, Precision, Recall, F1Score
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import SAMB
from automatic_mask_generator import SamAutomaticMaskGenerator
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric

from sklearn.metrics import jaccard_score, f1_score

def computeIoU(mask_pred, mask):
    return jaccard_score(mask.cpu().numpy(), mask_pred.cpu().numpy())

def computeDice(mask_pred, mask):
    return f1_score(mask.cpu().numpy(), mask_pred.cpu().numpy())

def computeHD(mask_pred, mask):
    return compute_hausdorff_distance(mask_pred.cpu().numpy(), mask.cpu().numpy())

def computeSensi(mask_pred, mask):
    TP = (mask_pred * mask).sum().item()  
    FN = ((1 - mask_pred) * mask).sum().item()  
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def computeSpeci(mask_pred, mask):
    TN = ((1 - mask_pred) * (1 - mask)).sum().item()  
    FP = (mask_pred * (1 - mask)).sum().item()  
    return TN / (TN + FP) if (TN + FP) > 0 else 0




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TNBC',type=str, help='TNBC')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    parser.add_argument('--model',default='../pretrain/sam_vit_b_01ec64.pth', type=str, help='')
    parser.add_argument('--prompt',default='auto', type=str, help='box, point, auto')
    args = parser.parse_args()
    
    save_png = f'visual/{args.dataset}/'

    os.makedirs(save_png,exist_ok=True)

    args.jsonfile = f'datasets/{args.dataset}/data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    test_files = df['test']
    test_dataset = BinaryLoader(args.dataset, test_files, A.Compose([
                                        A.Resize(args.size, args.size),
                                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensor()
                                        ]))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)
    

    model = SAMB(img_size=args.size)
    # 加载预训练权重
    pretrained_dict = torch.load(args.model)

# 获取当前模型的状态字典
    model_dict = model.state_dict()

# 找出不匹配的键
    unexpected_keys = [k for k in pretrained_dict.keys() if k not in model_dict]
    missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]

    print("Unexpected keys in state_dict:", unexpected_keys)
    print("Missing keys in state_dict:", missing_keys)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(args.model), strict=True)
    model.to(device='cuda')
    
    ###################################
    #computeIoU = 
    #computeHD = 
    #computeDice = 
    #computeSensi = 
    #computeSpeci = 

    avgIoU = []
    avgHD = []
    avgDice = []
    avgSensi = [] # sensitivity
    avgSpeci = [] # specificity

    ###################################


    mask_generator = SamAutomaticMaskGenerator(model)
    
    with torch.no_grad():
        for _, img, mask, img_id in tqdm(test_loader):

            if args.prompt == "auto":

                img_cv2 = cv2.imread(f'datasets/{args.dataset}/image_1024/{img_id[0]}.png')
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)         
                mask = Variable(mask).cuda()


            mask_list = mask_generator.generate(img_cv2)
            mask_pred = np.zeros((args.size, args.size))

            if len(mask_list) > 0:
            
                sorted_anns = sorted(mask_list, key=(lambda x: x['area']), reverse=True)

                # generate segmentation masks


                ###################################

            mask_pred = torch.tensor(mask_pred).cuda()
            mask_pred = mask_pred.unsqueeze(0).unsqueeze(0)

            mask_draw = mask_pred.clone().detach()
            gt_draw = mask.clone().detach()
            
            # calculate metrics
            iou = computeIoU(mask_pred, mask)
            dice = computeDice(mask_pred, mask)
            hd = computeHD(mask_pred, mask)
            sensitivity = computeSensi(mask_pred, mask)
            specificity = computeSpeci(mask_pred, mask)


            ###################################

            mask_pred = mask_pred.view(-1)
            mask = mask.view(-1)

            img_id = list(img_id[0].split('.'))[0]
            mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
            mask_numpy[mask_numpy==1] = 255 

            cv2.imwrite(f'{save_png}{img_id}.png',mask_numpy)
            
            image_ids = []
            avgDice.append(dice)
            avgIoU.append(iou)
            if hd != float("inf"):
                avgHD.append(hd)
            avgSensi.append(sensitivity)
            avgSpeci.append(specificity)
            image_ids.append(img_id)
            torch.cuda.empty_cache()
 
            
   


