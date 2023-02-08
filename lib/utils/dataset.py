import os
import cv2
import random
import torch
import fnmatch

import numpy as np
import torch.utils.data as data
import matplotlib.pylab as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from PIL import Image

class DataTransform(object):
    def __init__(self, scales, crop_size, is_disparity=False):
        self.scales = scales
        self.crop_size = crop_size
        self.is_disparity = is_disparity

    def __call__(self, data_dict):
        if type(self.scales) == tuple:
            # Continuous range of scales
            sc = np.random.uniform(*self.scales)

        elif type(self.scales) == list:
            # Fixed range of scales
            sc = random.sample(self.scales, 1)[0]

        raw_h, raw_w = data_dict['im'].shape[-2:]
        resized_size = [int(raw_h * sc), int(raw_w * sc)]
        i, j, h, w = 0, 0, 0, 0  # initialise cropping coordinates
        flip_prop = random.random()

        for task in data_dict:
            if len(data_dict[task].shape) == 2:   # make sure single-channel labels are in the same size [H, W, 1]
                data_dict[task] = data_dict[task].unsqueeze(0)

            # Resize based on randomly sampled scale
            if task in ['im']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.BILINEAR)
            elif task in ['depth', 'seg', 'disp']:
                data_dict[task] = transforms_f.resize(data_dict[task], resized_size, Image.NEAREST)

            # Add padding if crop size is smaller than the resized size
            if self.crop_size[0] > resized_size[0] or self.crop_size[1] > resized_size[1]:
                right_pad, bottom_pad = max(self.crop_size[1] - resized_size[1], 0), max(self.crop_size[0] - resized_size[0], 0)
                if task in ['im']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       padding_mode='reflect')
                elif task in ['seg', 'disp']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=-1, padding_mode='constant')  # -1 will be ignored in loss
                elif task in ['depth']:
                    data_dict[task] = transforms_f.pad(data_dict[task], padding=(0, 0, right_pad, bottom_pad),
                                                       fill=0, padding_mode='constant')  # 0 will be ignored in loss

            # Random Cropping
            if i + j + h + w == 0:  # only run once
                i, j, h, w = transforms.RandomCrop.get_params(data_dict[task], output_size=self.crop_size)
            data_dict[task] = transforms_f.crop(data_dict[task], i, j, h, w)

            # Random Flip
            if flip_prop > 0.5:
                data_dict[task] = torch.flip(data_dict[task], dims=[2])
           
            # Final Check:
            if task == 'depth':
                data_dict[task] = data_dict[task] / sc

            if task == 'disp':  # disparity is inverse depth
                data_dict[task] = data_dict[task] * sc

            if task == 'seg':
                data_dict[task] = data_dict[task].squeeze(0)
        return data_dict

class CityScapes(data.Dataset):
    """
    CityScapes dataset
    Included tasks:
        1. Semantic Segmentation,
        2. Disparity Estimation (Inverse Depth),
    """
    def __init__(self, root, train=True, augmentation=False):
        self.train = train
        self.root = os.path.expanduser(root)
        self.augmentation = augmentation

        # read the data file
        if train:
            self.data_path = root + '/train'
        else:
            self.data_path = root + '/val'

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + '/images'), '*.png'))
    
    def __len__(self):
        return self.data_len

    def decode_seg_map(self, mask):
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        mask_map = np.zeros_like(mask)
        mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = -1
        mask_map[np.isin(mask, [7])] = 0
        mask_map[np.isin(mask, [8])] = 1
        mask_map[np.isin(mask, [11])] = 2
        mask_map[np.isin(mask, [12])] = 3
        mask_map[np.isin(mask, [13])] = 4
        mask_map[np.isin(mask, [17])] = 5
        mask_map[np.isin(mask, [19])] = 6
        mask_map[np.isin(mask, [20])] = 7
        mask_map[np.isin(mask, [21])] = 8
        mask_map[np.isin(mask, [22])] = 9
        mask_map[np.isin(mask, [23])] = 10
        mask_map[np.isin(mask, [24])] = 11
        mask_map[np.isin(mask, [25])] = 12
        mask_map[np.isin(mask, [26])] = 13
        mask_map[np.isin(mask, [27])] = 14
        mask_map[np.isin(mask, [28])] = 15
        mask_map[np.isin(mask, [31])] = 16
        mask_map[np.isin(mask, [32])] = 17
        mask_map[np.isin(mask, [33])] = 18
        return mask_map
    
    def decode_disparity_map(self, disparity):
        # remap invalid points to -1 (not to conflict with 0, infinite depth, such as sky)
        disparity[disparity == 0] = -1
        # reduce by a factor of 4 based on the rescaled resolution
        disparity[disparity > -1] = (disparity[disparity > -1] - 1) / (256 * 4)
        return disparity
    
    def __getitem__(self, index):
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(plt.imread(self.data_path + '/images/{:d}.png'.format(index)), -1, 0)).float()
        disparity = cv2.imread(self.data_path + '/depth/{:d}.png'.format(index), cv2.IMREAD_UNCHANGED).astype(np.float32)
        disparity = torch.from_numpy(self.decode_disparity_map(disparity)).unsqueeze(0).float()
        seg = np.array(Image.open(self.data_path + '/seg/{:d}.png'.format(index)))
        seg = torch.from_numpy(self.decode_seg_map(seg)).unsqueeze(0).float()
    
        data_dict = {'im': image, 'seg': seg, 'disp': disparity}

        # apply data augmentation if required
        if self.augmentation:
            data_dict = DataTransform(crop_size=[256, 256], scales=[1.0])(data_dict)

        im = 2. * data_dict.pop('im') - 1.  # normalised to [-1, 1]
        return im, data_dict['seg'], data_dict['disp']
    
    

    
   