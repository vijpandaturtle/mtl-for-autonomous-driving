import timm
import torch 
import torch.nn as nn

from effdet.efficientdet import BiFpn
from effdet.config import fpn_config

from omegaconf import DictConfig
from lib.model.heads import BiFPNDecoder, SegmentationHead, DepthHead

class StandaloneConfig:
    image_size: tuple = (224, 224)
    min_level: int = 3
    max_level: int = 7
    num_levels: int = max_level - min_level + 1
    pad_type: str = ''  # use 'same' for TF style SAME padding
    act_type: str = 'silu'
    norm_layer = None  # defaults to batch norm when None
    norm_kwargs = dict(eps=.001, momentum=.01)
    separable_conv: bool = True
    apply_resample_bn: bool = True
    conv_after_downsample: bool = False
    conv_bn_relu_pattern: bool = False
    use_native_resize_op: bool = False
    downsample_type: bool = 'bilinear'
    upsample_type: bool = 'bilinear'
    redundant_bias: bool = False

    fpn_cell_repeats: int = 3
    fpn_channels: int = 88
    fpn_name: str = 'bifpn_fa'
    fpn_config: DictConfig = None  # determines FPN connectivity, if None, use default for type (name)

    def __post_init__(self):
        self.num_levels = self.max_level - self.min_level + 1


class DenseMultiNet(nn.Module):
    def __init__(self, config, backbone='convnext_atto', backbone_indices=(1, 2, 3)):
        super().__init__()
        
        self.backbone = timm.create_model(backbone, features_only=True, out_indices=backbone_indices, pretrained=True)
        self.neck = BiFpn(config, self.backbone.feature_info.get_dicts())
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=88)
       
        
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=19, #Semantic Segmentation Classes
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.part_segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1, #Part Segmentation Classes
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.depth_estimation_head = SegmentationHead(
            in_channels=64,
            out_channels=1, #Depth Classes
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

sc = StandaloneConfig()
data = torch.randn((3, 224, 224))
output1, output2, output3 = DenseMultiNet(sc, data)