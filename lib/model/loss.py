from copy import deepcopy
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

class SILogLoss(nn.Module):
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.
    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.
    Returns:
        float: SILog loss.
    """
    def __init__(self, variance_focus = 0.85):
        super(InvHuberLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, x, target):
        # let's only compute the loss on non-null pixels from the ground-truth depth-map
        non_zero_mask = (target > 0) & (x > 0)
        # SILog
        d = torch.log(x[non_zero_mask]) - torch.log(target[non_zero_mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def compute_loss(pred, gt, task_id):
    """
    Compute task-specific loss.
    """
    loss = None
    if task_id == 'semantic':
        # Cross Entropy Loss with Ignored Index (values are -1)
        #loss = F.nll_loss(pred, gt, ignore_index=-1)
        loss = F.cross_entropy(pred, gt, ignore_index=250)
        return loss 
    
    elif task_id == 'depth':
        # L1 Loss with Ignored Region (values are 0 or -1)
        invalid_idx = 250 if task_id == 'disp' else 0
        loss_fn = InvHuberLoss()
        valid_mask = (torch.sum(gt, dim=1, keepdim=True) != invalid_idx).to(pred.device)
        loss = torch.sum(loss_fn(pred, gt).masked_select(valid_mask)) \
                / torch.nonzero(valid_mask, as_tuple=False).size(0)
    return loss



