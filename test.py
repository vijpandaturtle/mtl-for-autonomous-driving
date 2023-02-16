import timm
import torch
from lib.multinet import DenseDrive
from pthflops import count_ops


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#4.2M parameters
#Without backbone, 0.86M trainable parameters

# Create a network and a corresponding input
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
backbone = timm.create_model('convnext_nano', features_only=True, out_indices=(0,1,2,3), pretrained=True)
mt_model = DenseDrive(backbone).to(device)
inp = torch.rand(3,256,512).to(device)

# Count the number of FLOPs
#print(count_ops(mt_model, inp))
print(count_parameters(mt_model))
