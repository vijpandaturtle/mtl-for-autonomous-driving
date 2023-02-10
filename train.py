import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.utils.data.sampler as sampler

from lib.model.multinet import DenseDrive
from lib.utils.dataset import CityScapes
from lib.model.metrics import ConfMatrix, depth_error
from lib.model.loss import compute_loss

import os
import fnmatch
import numpy as np
import random
import matplotlib.pyplot as plt

def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, total_epoch=200):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    
    avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
    
    for index in range(total_epoch):
        cost = np.zeros(12, dtype=np.float32)

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_depth = train_depth.to(device)
            train_label = train_label.squeeze(1).long()
         
            seg_pred, depth_pred = multi_task_model(train_data)
        
            optimizer.zero_grad()
            train_loss = [compute_loss(seg_pred, train_label, 'semantic'),
                          compute_loss(depth_pred, train_depth, 'depth')]
            print(train_loss)
          
            loss = sum([train_loss[i] for i in range(2)])
           
            loss.backward()
            optimizer.step()
           
            # accumulate label prediction for every pixel in training images
            conf_mat.update(seg_pred.argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(depth_pred, train_depth)
            avg_cost[index, :6] += cost[:6] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_depth = test_depth.to(device)
                test_label = train_label.squeeze(1).long()

                test_seg_pred, test_depth_pred = multi_task_model(test_data)
                test_loss = [compute_loss(test_seg_pred, test_label, 'semantic'),
                             compute_loss(test_depth_pred, test_depth, 'depth')]


                conf_mat.update(test_seg_pred.argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_depth_pred, test_depth)
                avg_cost[index, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 7:9] = conf_mat.get_metrics()

        scheduler.step()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mt_model = DenseDrive().to(device)
freeze_backbone = True
if freeze_backbone:
    mt_model.backbone.requires_grad_(False)
    #model.bifpn.requires_grad_(False)
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