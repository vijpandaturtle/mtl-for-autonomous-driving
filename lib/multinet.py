import torch 
import torch.nn as nn
import torch.nn.functional as F

from lib.blocks import inconv, down, up, outconv

class DenseDrive(nn.Module):
    def __init__(self):
        super(DenseDrive, self).__init__()
        self.seg_class_nb = 7

        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, False)
        self.up2 = up(512, 128, False)
        self.up3 = up(256, 64, False)
        self.up4 = up(128, 64, False)
        self.outc_segm = outconv(64, self.seg_class_nb)
        self.outc_depth = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2) 
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        seg = self.up1(x5, x4)
        seg = self.up2(seg, x3)
        seg = self.up3(seg, x2)
        seg = self.up4(seg, x1)
        seg_out = self.outc_segm(seg)
        seg_out = F.log_softmax(seg_out, dim=1)

        depth = self.up1(x5, x4)
        depth = self.up2(depth, x3)
        depth = self.up3(depth, x2)
        depth = self.up4(depth, x1)
        depth_out = self.outc_depth(depth)
        depth_out = torch.sigmoid(depth_out) 
        
        return seg_out, depth_out  