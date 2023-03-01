import os
import fnmatch
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class RandomScaleCrop(object):
    def __init__(self, scale=[1.0, 1.2, 1.5]):
        self.scale = scale

    def __call__(self, img, label, depth):
        height, width = img.shape[-2:]
        sc = self.scale[random.randint(0, len(self.scale)-1)]
        h, w = int(height/sc), int(width/sc)
        i = random.randint(0, height - h)
        j = random.randint(0, width - w)
        img_ = F.interpolate(img[None,:,i:i+h, j:j+w], size=(height, width), mode='bilinear').squeeze(0)
        label_ = F.interpolate(label[None,None,i:i+h, j:j+w], size=(height, width), mode='nearest').squeeze(0).squeeze(0)
        depth_ = F.interpolate(depth[None,:,i:i+h, j:j+w], size=(height, width), mode='nearest').squeeze(0)
        depth_ = depth_ / sc
        return img_, label_, depth_

class CityScapes(Dataset):
    def __init__(self, root, train=True, transforms=None, random_flip=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.transform = transforms
        self.random_flip = random_flip

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/image'), '*.npy'))

    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0)).float()
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index))).float()
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0)).float()
        #depth = np.minimum(depth, 100)

        if self.transform is not None:
            image, semantic, depth = self.transform(image, semantic, depth)
        if self.random_flip and torch.rand(1)<0.5:
            image = torch.flip(image, dims=[2])
            semantic = torch.flip(semantic, dims=[1])
            depth = torch.flip(depth, dims=[2])

        return image, semantic, depth

    def __len__(self):
        return self.data_len