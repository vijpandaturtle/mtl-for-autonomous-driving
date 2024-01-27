import random

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data.dataset import Dataset

from lib.dataset import CityScapes, RandomScaleCrop
from lib.efficientmtl import EfficientMTL
from lib.trainer import multi_task_trainer

sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "total_loss"},
    "parameters": {
        "alpha": {"values": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]},
    },
}


def main():
    wandb.init(project="convmtl-search")

    config = wandb.config
    config.random_seed = 54321  # or any of your favorite number
    config.lr = 1e-3
    config.lr_weight_decay = 1e-6
    config.epochs = 50
    config.train_batch_size = 4
    config.val_batch_size = 4
    config.alpha = 0.25
    config.beta = 1 - config.alpha
    config.t_0 = 50
    config.t_mult = 1
    config.eta_min = 1e-6

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.random_seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone = timm.create_model(
        "convnext_tiny", features_only=True, out_indices=(0, 1, 2, 3), pretrained=True
    )
    mt_model = EfficientMTL(backbone).to(device)

    optimizer = optim.AdamW(
        mt_model.parameters(), lr=config.lr, weight_decay=config.lr_weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.t_0,  # Number of iterations for the first restart
        T_mult=config.t_mult,  # A factor increases TiTi​ after a restart
        eta_min=config.eta_min,
    )  # Minimum learning rate

    freeze_backbone = True
    if freeze_backbone:
        mt_model.backbone.requires_grad_(False)
        print("[Info] freezed backbone")

    # for param in mt_model.backbone['stages_3'].parameters():
    #     param.requires_grad = True

    print(
        "LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR <11.25 <22.5"
    )

    dataset_path = "cityscapes_processed"
    train_set = CityScapes(
        root=dataset_path, train=True, transforms=RandomScaleCrop(), random_flip=False
    )
    test_set = CityScapes(root=dataset_path, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config.train_batch_size,
        drop_last=True,  # difference in no of samples in last batch
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=config.val_batch_size,
        drop_last=True,
        shuffle=False,
    )

    multi_task_trainer(
        train_loader,
        test_loader,
        mt_model,
        device,
        optimizer,
        scheduler,
        config.epochs,
        config.alpha,
        config.beta,
    )

    # model_path = dataset_path + '/efficientmtl_weights.pt'
    # full_model_path = dataset_path + '/efficientmtl_checkpoint.pt'
    # #torch.save(the_model.state_dict(), PATH)
    # torch.save(mt_model, model_path)
    # state = {
    #     'epoch': config.epochs,
    #     'state_dict': mt_model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # torch.save(state, full_model_path)

    # art = wandb.Artifact("efficientmtl", type="model")
    # art.add_file("efficientmtl_weights.pt")
    # wandb.log_artifact(art)


sweep_id = wandb.sweep(sweep=sweep_configuration, project="convmtl-search")
count = 10
wandb.agent(sweep_id=sweep_id, function=main, count=count)
# main()

# Reimporting SSL ResNet for multitask learning
# import pathlib
# from timm.models.resnet import default_cfgs
# model_name = "resnet50"
# checkpoint_path = "model.pth"
# checkpoint_path_url = pathlib.Path(checkpoint_path).resolve().as_uri()

# default_cfgs["url"] = checkpoint_path_url

# model = timm.create_model(
#     model_name,
#     features_only=True,
#     out_indices=(1,2,3,4)
# )
