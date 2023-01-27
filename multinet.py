import torch 
import torch.nn as nn 

class DenseMultiNet(nn.Module):
    def __init__(self, config, backbone='convnext_atto', backbone_indices=(1, 2, 3)):
        super(PercepMultiNet).__init__()
        
        self.backbone = timm.create_model(backbone, features_only=True, out_indices=backbone_indices, pretrained=True)
        self.neck = BiFpn(config, self.backbone.feature_info.get_dicts())
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters[self.compound_coef])
        
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes+1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.part_segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes+1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.depth_estimation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes+1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        p2, p3, p4, p5 = self.encoder(inputs)[-4:]  # self.backbone_net(inputs)

        features = (p3, p4, p5)

        features = self.bifpn(features)
        
        p3,p4,p5,p6,p7 = features
        
        outputs = self.bifpndecoder((p2,p3,p4,p5,p6,p7))

        semantic_seg_map = self.segmentation_head(outputs)
        part_seg_map = self.part_segmentation_head(outputs)
        depth_map = self.depth_estimation_head(outputs)

        return semantic_seg_map, part_seg_map, depth_map
