from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

class InvHuberLoss(nn.Module):
    """Inverse Huber Loss for depth estimation.
    The setup is taken from https://arxiv.org/abs/1606.00373
    Args:
      ignore_index (float): value to ignore in the target
                            when computing the loss.
    """
    def __init__(self, ignore_index=0):
        super(InvHuberLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, target):
        input = F.relu(x)  # depth predictions must be >=0
        diff = input - target
        mask = target != self.ignore_index

        err = torch.abs(diff * mask.float())
        c = 0.2 * torch.max(err)
        err2 = (diff ** 2 + c ** 2) / (2.0 * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = torch.mean(err * mask_err.float() + err2 * mask_err2.float())
        return cost    

def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    if task_id in ['seg', 'instance_seg', 'part_seg'] or 'class' in task_id:
        # Cross Entropy Loss with Ignored Index (values are -1)
        loss = F.cross_entropy(pred, gt, ignore_index=-1)

    if task_id in ['depth', 'disp']:
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = -1 if task_id == 'disp' else 0
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(F.l1_loss(pred, gt, reduction='none').masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss


