import torch
import torch.nn as nn

class ConditionLoss(nn.Module):
    def __init__(self):
        super(ConditionLoss, self).__init__()

    def forward(self, pred_cls, gt_cls, pred_bbox, gt_bbox):
        batch_loss = 0.0
        for i in range(len(gt_cls)):
            print(gt_cls[i])
            if int(gt_cls[i].item()) == 1:
                batch_loss += nn.BCELoss(pred_cls[i], gt_cls[i])# + nn.MSELoss(pred_bbox[i], gt_bbox[i])
            if int(gt_cls[i].item()) == 0:
                batch_loss += nn.BCELoss(pred_cls[i], gt_cls[i])

        batch_loss /= len(gt_cls)

        return batch_loss