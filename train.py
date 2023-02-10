import timm
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from lib.utils.dataset import CityScapes
from lib.model.multinet import DenseDrive
from lib.trainer import multi_task_trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model('convnext_atto', features_only=True, out_indices=(0,1,2,3), pretrained=True)
mt_model = DenseDrive(backbone).to(device)

freeze_backbone = True
if freeze_backbone:
    mt_model.backbone.requires_grad_(False)
    print('[Info] freezed backbone')

# if opt.freeze_seg:
#     model.bifpndecoder.requires_grad_(False)
#     model.segmentation_head.requires_grad_(False)
#     print('[Info] freezed segmentation head')


optimizer = optim.Adam(mt_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')

dataset_path = 'cityscapes'
train_set = CityScapes(root=dataset_path, train=True)
test_set = CityScapes(root=dataset_path, train=False)

batch_size = 25
train_loader = torch.utils.data.DataLoader(
               dataset=train_set,
               batch_size=batch_size,
               drop_last=True, #difference in no of samples in last batch
               shuffle=True)

test_loader = torch.utils.data.DataLoader(
              dataset=test_set,
              batch_size=batch_size,
              drop_last=True,
              shuffle=False)

multi_task_trainer(train_loader, test_loader, mt_model, device, optimizer, scheduler, 100)