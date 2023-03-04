import timm
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.dataset import Dataset

from lib.dataset import CityScapes, RandomScaleCrop
from lib.multinet import DenseDrive
from lib.trainer import multi_task_trainer

random_seed = 54321 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model('efficientnet_b0', features_only=True, out_indices=(1,2,3,4), pretrained=True)
mt_model = DenseDrive(backbone).to(device)

freeze_backbone = True
if freeze_backbone:
    mt_model.backbone.requires_grad_(False)
    print('[Info] freezed backbone')

# Unfreeze the last stage
for param in mt_model.stages[3].parameters():
    param.requires_grad = True


optimizer = optim.AdamW(mt_model.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                        T_0 = 8, # Number of iterations for the first restart
                                        T_mult = 1, # A factor increases TiTi​ after a restart
                                        eta_min = 1e-4) # Minimum learning rate

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')

dataset_path = 'cityscapes_processed'
train_set = CityScapes(root=dataset_path, train=True, transforms=RandomScaleCrop(), random_flip=False)
test_set = CityScapes(root=dataset_path, train=False)

epochs = 200
#batch_size = 16
train_loader = torch.utils.data.DataLoader(
               dataset=train_set,
               batch_size=16,
               drop_last=True, #difference in no of samples in last batch
               shuffle=True)

test_loader = torch.utils.data.DataLoader(
              dataset=test_set,
              batch_size=32,
              drop_last=True,
              shuffle=False)

multi_task_trainer(train_loader, test_loader, mt_model, device, optimizer, scheduler, epochs)

model_path = dataset_path + '/densedrive_femto_v0.pt'
#torch.save(the_model.state_dict(), PATH)
torch.save(mt_model, model_path)