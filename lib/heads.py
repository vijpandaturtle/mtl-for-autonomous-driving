import torch 
import torch.nn as nn 
import torch.nn.functional as F
from lib.blocks import SegmentationBlock, MergeBlock, Activation

class BiFPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_depth=5,
            pyramid_channels=128,
            segmentation_channels=128,
            dropout=0.4,
            merge_policy="add",):
        super().__init__()

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [5, 4, 3, 2, 1]
        ])
        
        self.seg_p2 = SegmentationBlock(96, 128, n_upsamples=0)
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, inputs):
        p2, p3, p4, p5, p6, p7 = inputs
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p7, p6, p5, p4, p3])]
        p2 = self.seg_p2(p2)
            
        p3,p4,p5,p6,p7 = feature_pyramid
        x = self.merge((p2,p3,p4,p5,p6,p7))
        x = self.dropout(x)
        
        return x


class DepthwiseASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(DepthwiseASPP, self).__init__()
        
        # Depthwise separable convolutions with atrous rates
        self.conv1_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=atrous_rates[0], dilation=atrous_rates[0], groups=in_channels)
        self.conv1_pointwise = nn.Conv2d(in_channels, 256, kernel_size=1)
        
        self.conv2_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=atrous_rates[1], dilation=atrous_rates[1], groups=in_channels)
        self.conv2_pointwise = nn.Conv2d(in_channels, 256, kernel_size=1)
        
        self.conv3_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=atrous_rates[2], dilation=atrous_rates[2], groups=in_channels)
        self.conv3_pointwise = nn.Conv2d(in_channels, 256, kernel_size=1)
        
        self.conv4_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=atrous_rates[3], dilation=atrous_rates[3], groups=in_channels)
        self.conv4_pointwise = nn.Conv2d(in_channels, 256, kernel_size=1)
        
    def forward(self, x):
        feat1 = self.conv1_depthwise(x)
        feat1 = self.conv1_pointwise(feat1)
        
        feat2 = self.conv2_depthwise(x)
        feat2 = self.conv2_pointwise(feat2)
        
        feat3 = self.conv3_depthwise(x)
        feat3 = self.conv3_pointwise(feat3)
        
        feat4 = self.conv4_depthwise(x)
        feat4 = self.conv4_pointwise(feat4)
        
        # Concatenate and return the output
        x = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        return x

class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(DeepLabV3Head, self).__init__()
        
        # Atrous spatial pyramid pooling
        self.aspp = DepthwiseASPP(in_channels, [6, 12, 18, 24])
        
        # 1x1 convolution for feature fusion
        self.conv1 = nn.Conv2d(in_channels + 256, 256, kernel_size=1)
        
        # Final convolution layer for semantic segmentation
        self.conv2 = nn.Conv2d(256, out_channels, kernel_size=1)

        if activation == "logsoftmax":
            self.activation = nn.LogSoftmax()
        else:
            self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # Atrous spatial pyramid pooling
        x = self.aspp(x)
        
        # Feature fusion
        x = F.interpolate(x, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, x], dim=1)
        x = self.conv1(x)
        
        # Semantic segmentation
        x = self.conv2(x)
        x = F.interpolate(x, size=(x.size(2)*4, x.size(3)*4), mode='bilinear', align_corners=False)
        x = self.activation(x)

        return x

# class SegmentationHead(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         activation = Activation(activation)
#         super().__init__(conv2d, upsampling, activation)

# class DepthHead(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
#         conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
#         upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
#         activation = Activation(activation)
#         super().__init__(conv2d, upsampling, activation)
