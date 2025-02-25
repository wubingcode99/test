# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3)
from .conv import (CBAM, EMA_attention, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)
from .carafe import CARAFE
from .dysample import DySample
from ultralytics.nn.modules.msda import MultiDilatelocalAttention
from ultralytics.nn.modules.C2f_faster_msda import C2f_Faster_Msda
from .MLCA import MLCA
from .C2f_mlca import C2f_mlca
from .C2f_DLKA import C2f_DLKA, deformable_LKA_Attention
from .C2f_MSDA import C2f_MSDA
from .EMA import EMA
from .MCA import MCA
from .ca import CA_Block
from .DCNV2 import C2f_DCNv2
from .C2f_DCN import C2f_DCN
from .C2f_SimAM import C2f_SimAM
from .Biformer import BiLevelRoutingAttention
from .SPPF_GMA import SPPF_EMA
from ultralytics.nn.modules.sa import SelfAttention
from .AKConv import AKConv
from .SCConv import SCConv_yolov8

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'EMA_attention', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'CARAFE',
           'MultiDilatelocalAttention', 'C2f_Faster_Msda', 'MLCA', 'C2f_mlca', 'C2f_DLKA', 'C2f_MSDA',
           'deformable_LKA_Attention', 'EMA', 'CA_Block', 'MCA', 'C2f_DCNv2', 'C2f_SimAM', 'BiLevelRoutingAttention',
           'C2f_DCN', 'SPPF_EMA','SelfAttention', 'AKConv', 'SCConv_yolov8')
