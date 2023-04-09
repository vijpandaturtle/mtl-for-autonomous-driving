import wandb
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

##############################
wandb.init(project="densedrive")

config = wandb.config
config.random_seed =  54321 # or any of your favorite number 
config.backbone_name = 'efficientnet_b4'
config.lr = 9e-4
config.lr_weight_decay = 1e-6
config.epochs = 100
config.train_batch_size = 4
config.val_batch_size = 4
config.t_0 = 30
config.t_mult = 2
config.eta_min = 1e-5

##############################
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed(config.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config.random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model('efficientnet_b4', features_only=True, out_indices=(1,2,3,4), pretrained=True)
mt_model = DenseDrive(backbone).to(device)

optimizer = optim.AdamW(mt_model.parameters(), lr=config.lr, weight_decay=config.lr_weight_decay)
scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                        T_0 = config.t_0, # Number of iterations for the first restart
                                        T_mult = config.t_mult, # A factor increases TiTi​ after a restart
                                        eta_min = config.eta_min) # Minimum learning rate

freeze_backbone = True
if freeze_backbone:
    mt_model.backbone.requires_grad_(False)
    print('[Info] freezed backbone')

for param in mt_model.backbone.blocks[-4:].parameters():
    param.requires_grad = True

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5')

dataset_path = 'cityscapes_processed'
train_set = CityScapes(root=dataset_path, train=True, transforms=RandomScaleCrop(), random_flip=False)
test_set = CityScapes(root=dataset_path, train=False)

train_loader = torch.utils.data.DataLoader(
               dataset=train_set,
               batch_size=config.train_batch_size,
               drop_last=True, #difference in no of samples in last batch
               shuffle=True)

test_loader = torch.utils.data.DataLoader(
              dataset=test_set,
              batch_size=config.val_batch_size,
              drop_last=True,
              shuffle=False)

multi_task_trainer(train_loader, test_loader, mt_model, device, optimizer, scheduler, config.epochs)

#####################################
model_path = dataset_path + '/densedrive_efficientnet_b0_final.pt'
full_model_path = dataset_path + '/densedrive_efficientnet_b0_checkpoint.pt'
#torch.save(the_model.state_dict(), PATH)
torch.save(mt_model, model_path)
state = {
    'epoch': config.epochs,
    'state_dict': mt_model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, full_model_path)