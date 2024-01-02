import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision.ops import generalized_box_iou_loss, distance_box_iou_loss, complete_box_iou_loss
import torchvision.transforms as transforms
import random
import os
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm

from ReadVideo import *
from my_dataset import Image_Kpts_Dataset
from model_lighten import Bbox_Lighten_Predictor


myseed = 0
random.seed(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
    torch.cuda.manual_seed(myseed)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


video_path = r'E:\DLCV_vq2d_data\clips'
train_json_file = r'E:\LoFTR\data\train_pos_neg_info.json'
val_json_file = r'E:\LoFTR\data\val_pos_neg_info.json'
save_model_path = r'E:\LoFTR\save_models'
save_model_name = 'model_resnet50.pth'
os.makedirs(save_model_path, exist_ok=True)

learning_rate = 1e-4
batch_size = 8
epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_dataset = Image_Kpts_Dataset(video_path=video_path, json_file=train_json_file, transform=transform)
val_dataset   = Image_Kpts_Dataset(video_path=video_path, json_file=val_json_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

model = Bbox_Lighten_Predictor()
model.load_state_dict(torch.load(r'E:\LoFTR\save_models\15000_best_model_resnet50.pth', map_location='cpu'))
model.to(device)
print(sum(p.numel() for p in model.parameters()))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

bbox_criterion = nn.SmoothL1Loss()
bbox_giou = generalized_box_iou_loss
cls_criterion  = nn.BCELoss()

minLoss = 1000.
validation_interval = 5000
k = 0.1
for epoch in range(1, epochs):
    ### Train ###
    k = min(k, 0.3)
    model.train()
    train_loss_batch = 0.0
    with tqdm(total=len(train_loader)) as train_bar:
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            crop_image, target_frame, kpts, gt_bbox, gt_label = batch
            pred = model(crop_image.to(device), target_frame.to(device), kpts.float().to(device))
            pred['bbox'] = (pred['bbox'] * gt_label.to(device).view(gt_label.shape[0],1).repeat(1,4))
            
            ### Define loss function ###
            loss_p = cls_criterion(pred['cls'].flatten(), gt_label.float().to(device))
            loss_l1 = bbox_criterion(pred['bbox'], gt_bbox.float().to(device))
            loss_giou = k * bbox_giou(pred['bbox'], gt_bbox.float().to(device), reduction='mean')
            if loss_giou < 0:
                loss_giou = 0
            loss = loss_p + loss_l1 + loss_giou

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_bar.set_postfix({
                'Loss_p': loss_p.item(),
                'Loss_L1': loss_l1.item(),
                'Loss_GIoU': loss_giou.item(),
                'Train Loss': loss.item()},
                refresh=True)
            train_loss_batch += loss.detach().cpu().item()

            if (batch_idx + 1) % validation_interval == 0:
                print(f"----Evaluation at {batch_idx + 1}----")
                model.eval()
                with tqdm(total=len(val_loader)) as val_bar:
                    with torch.no_grad():
                        for batch in tqdm(val_loader):
                            crop_image, target_frame, kpts, gt_bbox, gt_label = batch
                            pred = model(crop_image.to(device), target_frame.to(device), kpts.float().to(device))
                            pred['bbox'] = (pred['bbox'] * gt_label.to(device).view(gt_label.shape[0],1).repeat(1,4))
                            # print('Pred', pred['bbox'])
                            # print('GT', gt_bbox)

                            ### Define loss function ###
                            loss_p = cls_criterion(pred['cls'].flatten(), gt_label.float().to(device))
                            loss_l1 = bbox_criterion(pred['bbox'], gt_bbox.float().to(device))
                            loss_giou = k * bbox_giou(pred['bbox'], gt_bbox.float().to(device), reduction='mean')
                            if loss_giou < 0:
                                loss_giou = 0
                            loss = loss_p + loss_l1 + loss_giou
                            
                            val_bar.set_postfix({
                                'Loss_p': loss_p.item(),
                                'Loss_L1': loss_l1.item(),
                                'Loss_GIoU': loss_giou.item(),
                                'Val Loss': loss.item()},
                                refresh=True)
                            torch.save(model.state_dict(), os.path.join(save_model_path, f'{epoch+1}_{batch_idx + 1}_best_{save_model_name}'))
    
    train_loss_epoch = train_loss_batch / len(train_loader)
    torch.save(model.state_dict(), os.path.join(save_model_path, f'{epoch + 1}_{save_model_name}'))
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_epoch}')
    k += 0.1

    ### Evaluation ###
    # val_loss_batch = 0.0
    # model.eval()
    # with tqdm(total=len(val_loader)) as val_bar:
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             crop_image, target_frame, kpts, gt_bbox, gt_label = batch
    #             pred = model(crop_image.to(device), target_frame.to(device), kpts.float().to(device))
    #             print('GT', gt_bbox)
    #             pred['bbox'] = (pred['bbox'] * gt_label.to(device).view(gt_label.shape[0],1).repeat(1,4))
    #             print('Pred', pred['bbox'])
                
    #             k = 0.3
    #             loss_p = cls_criterion(pred['cls'].flatten(), gt_label.float().to(device))
    #             loss_l1 = bbox_criterion(pred['bbox'], gt_bbox.float().to(device))
    #             # loss_diou = k * bbox_diou(pred['bbox'], gt_bbox.float().to(device), reduction='mean')
    #             # if loss_diou < 0:
    #             #     loss_diou = 0
    #             loss = loss_p  + loss_l1
    
    #         torch.save(model.state_dict(), os.path.join(save_model_path, f'{epoch+1}_{save_model_name}'))

    #         if loss.item() < minLoss:
    #             minLoss = loss.item()
    #             torch.save(model.state_dict(), os.path.join(save_model_path, f'best_{save_model_name}'))
            
    #         val_bar.set_postfix({
    #             'Loss_p': loss_p.item(),
    #             'Loss_L1': loss_l1.item(),
    #             # 'Loss_DIoU': loss_diou.item(),
    #             'Val Loss': loss.item()},
    #             refresh=True)
    #         val_loss_batch += loss.detach().cpu().item()
    
    # val_loss_epoch = val_loss_batch / len(val_loader)
    # print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss_epoch}')


