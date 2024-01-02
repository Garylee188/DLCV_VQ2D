import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import json
import numpy as np
from PIL import Image
from ReadVideo import *
from kpts_preprocessing import point_preprocess

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def rescale_pts(points, old_shape, new_shape=(224, 224)):
    new_height, new_width = new_shape
    old_height, old_width = old_shape
    resized_points = [(int(x * new_width / old_width), int(y * new_height / old_height)) for x, y in points]
    return np.array(resized_points)

class Image_Kpts_Dataset(Dataset):
    def __init__(self, video_path, json_file, transform=None):
        self.video_path = video_path
        self.transform = transform
        with open(json_file, 'r') as file:
            self.data = json.load(file)
    
    def __getitem__(self, index):
        data_info = self.data[index]
        query_fnum = data_info["query_fnum"]
        visual_crop_info = data_info["visual_crop"]
        fnum = data_info["fnum"]
        pred_kpts = data_info["pred_kpts"]
        gt_bbox = data_info["gt_bbox"]
        origin_h = int(visual_crop_info["original_height"])
        origin_w = int(visual_crop_info["original_width"])
        
        ### bbox
        if gt_bbox == [0, 0, 0, 0]:
            bbox = np.array([0, 0, 0, 0])
            label = 0
        else:
            xmin = int(gt_bbox["x"])
            ymin = int(gt_bbox["y"])
            xmax = int(gt_bbox["x"]) + int(gt_bbox["width"])
            ymax = int(gt_bbox["y"]) + int(gt_bbox["height"])
            bbox = rescale_pts([(xmin, ymin), (xmax, ymax)], old_shape=(origin_h, origin_w))
            bbox = bbox.reshape(-1) / 224
            label = 1
        
        ### target_frame, visual_crop
        vd = ReadVideo(os.path.join(self.video_path, data_info["uid"] + ".mp4"))
        frames = vd.get_batch([i for i in range(query_fnum + 1)] + \
                               [visual_crop_info["frame_number"]]).asnumpy() 
        target_frame = frames[fnum]
        query_frame = frames[-1]
        left = int(visual_crop_info["x"])
        right = int(visual_crop_info["x"]) + int(visual_crop_info["width"])
        up = int(visual_crop_info["y"])
        down = int(visual_crop_info["y"]) + int(visual_crop_info["height"])
        crop_image = query_frame[up:down, left:right, :]

        target_frame = self.transform(Image.fromarray(target_frame))
        crop_image = self.transform(Image.fromarray(crop_image))

        ### kpts
        kpts = rescale_pts(point_preprocess(pred_kpts), old_shape=(origin_h, origin_w))
        kpts = kpts.T

        return crop_image, target_frame, kpts, bbox, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    with open(r'E:\LoFTR\data\60_train_pos_neg_info.json', 'r') as file:
        train_json = json.load(file)
    dataset = Image_Kpts_Dataset(video_path=r'E:\DLCV_vq2d_data\clips', json_file=r'E:\LoFTR\data\60_train_pos_neg_info.json', transform=transform)
    for i in range(10):
        crop_image, target_frame, kpts, bbox, label = dataset.__getitem__(i)
        print(bbox)
