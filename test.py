import torch
from lib.model.multinet import DenseDrive
from lib.model.segnet import SegNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = DenseDrive()
model.backbone.requires_grad_(False)
print(count_parameters(model))
#4.2M parameters
#Without backbone, 0.86M trainable parameters

