import torch 
import torch.nn as nn
import torch.nn.functional as F

from lib.neck import BiFPN
from lib.heads import BiFPNDecoder, SegmentationHead, DepthHead

class DenseDrive(nn.Module):
    def __init__(self, backbone):
        super(DenseDrive, self).__init__()

        self.fpn_num_filters = 64
        self.fpn_cell_repeats = 3
        self.conv_channels = [40, 112, 320]
        self.seg_class_nb = 7
        
        self.backbone = backbone
        
        self.neck = nn.Sequential(
            *[BiFPN(self.fpn_num_filters, self.conv_channels,
                    True if _ == 0 else False,
                    attention=True,
                    use_p8=False)
              for _ in range(self.fpn_cell_repeats)]
        )

        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters)
       
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=self.seg_class_nb, #Semantic Segmentation Classes
            activation='logsoftmax',
            kernel_size=1,
            upsampling=4,
        )

        self.depth_estimation_head = DepthHead(
            in_channels=64,
            out_channels=1, #Depth Classes
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.initialize_decoder(self.bifpndecoder)
        self.initialize_head(self.segmentation_head)
        self.initialize_head(self.depth_estimation_head)
        self.initialize_decoder(self.neck)

    def forward(self, x):
        p2, p3, p4, p5 = self.backbone(x)[-4:]
       
        features = (p3, p4, p5)
        features = self.neck(features)
     
        p3,p4,p5,p6,p7 = features

        outputs = self.bifpndecoder((p2,p3,p4,p5,p6,p7))

        semantic_seg_map = self.segmentation_head(outputs)
        depth_map = self.depth_estimation_head(outputs)
        return semantic_seg_map, depth_map

    def initialize_decoder(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


