import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            # elif pool_type=='lse':
            #     # LSE pool only
            #     lse_pool = logsumexp_2d(x)
            #     channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3)
        return scale
    
class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.Q = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.Q[1].weight, 0)
        nn.init.constant_(self.Q[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        
        self.ChannelGate = ChannelGate(self.in_channels)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)


        
    def forward(self, detect, aim):

        

        batch_size, channels, height_a, width_a = aim.shape
        batch_size, channels, height_d, width_d = detect.shape


        #####################################find aim image similar object ####################################################

        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()

        a_x = self.g(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, self.inter_channels, height_a, width_a)
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim

        non_det = torch.matmul(fi_div_C, a_x)
        non_det = non_det.permute(0, 2, 1).contiguous()
        non_det = non_det.view(batch_size, self.inter_channels, height_d, width_d)
        non_det = self.Q(non_det)
        non_det = non_det + detect

        ##################################### Response in chaneel weight ####################################################

        c_weight = self.ChannelGate(non_aim)
        act_aim = non_aim * c_weight
        act_det = non_det * c_weight

        return non_det, act_det, act_aim, c_weight

class Bbox_Lighten_Predictor(nn.Module):

    def __init__(self):
        super(Bbox_Lighten_Predictor, self).__init__()
        
        self.resnet50 = torch.nn.Sequential(*(list(resnet50(weights='DEFAULT').children())[:-1]))
        self.match_block = match_block(inplanes=2048)

        self.fc_concat = nn.Sequential(
            nn.Linear(2048+2048+256, 1024),
            nn.ReLU()
        )
        self.fc_ = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.fc_reg = nn.Linear(256, 4)  # Output 4 values for bounding box (xmin, ymin, xmax, ymax)
        self.fc_cls = nn.Linear(256, 1)  # Output 1 values for positive/negative sample (1/0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, crop, frame, kpts):
        frame_feat = self.resnet50(frame)
        crop_feat = self.resnet50(crop)
        non_det, act_det, act_aim, c_weight = self.match_block(frame_feat, crop_feat)
        act_det = act_det.view(act_det.size(0), -1)
        act_aim = act_aim.view(act_aim.size(0), -1)
        kpts_feat = kpts.view(kpts.size(0), -1)

        out = self.fc_concat(torch.cat([act_aim, act_det, kpts_feat], dim=-1))
        out = self.fc_(out)

        bbox_out = self.sigmoid(self.fc_reg(out))
        cls_out = self.sigmoid(self.fc_cls(out))
        return {
            'bbox': bbox_out,
            'cls': cls_out
        }

if __name__ == '__main__':
    device = 'cuda'
    bbox_predictor = Bbox_Lighten_Predictor()
    bbox_predictor.to(device)
    bbox_predictor.train()
    N = 2
    dummy_crop = torch.randn(N, 3, 224, 224)
    dummy_frame = torch.randn(N, 3, 224, 224)
    dummy_keypoints = torch.randn(N, 2, 128)
    # print(dummy_keypoints)

    result = bbox_predictor(dummy_crop.to(device), dummy_frame.to(device), dummy_keypoints.to(device))
    print(result['bbox'].size())
    print(result['cls'].size())
    # result = bbox_predictor(dummy_crop.to(device), dummy_frame.to(device), dummy_keypoints.to(device))
    # print(result['bbox'])
    # print(result['cls'])