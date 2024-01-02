import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision.transforms as transforms
import random
import os
import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ReadVideo import *
from my_dataset import Image_Kpts_Dataset, rescale_pts
from model_lighten import Bbox_Lighten_Predictor


def plot_bbox(frame, bbox):
    bbox *= 224
    # bbox = rescale_pts([(bbox[0], bbox[1]), (bbox[2], bbox[3])], old_shape=(224,224), new_shape=(360, 640))
    # bbox = bbox.reshape(-1)

    # frame = transforms.Resize((360, 640))(frame)
    frame_np = frame.numpy()
    ### patches.Rectangle((xmin, ymin), w, h)  ###
    bbox_rect = patches.Rectangle(
        (bbox[0], bbox[1]), 
        bbox[2] - bbox[0], 
        bbox[3] - bbox[1], 
        linewidth=2, edgecolor='r', facecolor='none'
    )
    fig, ax = plt.subplots(1)
    ax.imshow(frame_np.transpose((1, 2, 0)))
    ax.add_patch(bbox_rect)
    plt.show()

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
save_model_path = r'E:\LoFTR\save_models\1000_best_model_resnet50.pth'

batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_dataset = Image_Kpts_Dataset(video_path=video_path, json_file=train_json_file, transform=transform)
val_dataset   = Image_Kpts_Dataset(video_path=video_path, json_file=val_json_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Bbox_Lighten_Predictor()
checkpoint = torch.load(save_model_path, map_location='cpu')
# print(checkpoint.keys())
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
model.to(device)

### Inference ###
model.eval()
with torch.no_grad():
    for batch in val_loader:
        crop_image, target_frame, kpts, gt_bbox, gt_label = batch
        print(kpts.shape)
        pred = model(crop_image.to(device), target_frame.to(device), kpts.float().to(device))
        # print(pred['cls'])
        if pred['cls'].item() >= 0.5:
            plot_bbox(target_frame.cpu().squeeze(), pred['bbox'].cpu().squeeze())
