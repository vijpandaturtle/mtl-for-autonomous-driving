import timm
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from lib.dataset import CityScapes
from lib.multinet import DenseDrive
from lib.trainer import multi_task_trainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model('convnext_femto', features_only=True, out_indices=(0,1,2,3), pretrained=False)
mt_model = DenseDrive(backbone).to(device)

freeze_backbone = False
if freeze_backbone:
    mt_model.backbone.requires_grad_(False)
    print('[Info] freezed backbone')

# if opt.freeze_seg:
#     model.bifpndecoder.requires_grad_(False)
#     model.segmentation_head.requires_grad_(False)
#     print('[Info] freezed segmentation head')


optimizer = optim.Adam(mt_model.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')

dataset_path = 'cityscapes_processed'
train_set = CityScapes(root=dataset_path, train=True)
test_set = CityScapes(root=dataset_path, train=False)

epochs = 25
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

multi_task_trainer(train_loader, test_loader, mt_model, device, optimizer, scheduler, 25)

model_path = dataset_path + '/model/densedrive_femto_v0.pt'.
#torch.save(the_model.state_dict(), PATH)
torch.save(mt_model, model_path)