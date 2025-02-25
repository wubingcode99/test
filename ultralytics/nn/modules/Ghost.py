import math
import torch
import torch.nn as nn
class GhostModule(nn.Module):
    # inputs: 输入特征图
    # outputs_channel: 该模块的输出通道数
    # kernel: 深度卷积下采样时所需的卷积核尺寸
    # strides: 是否需要下采样
    # exp_channel: Ghost模块的输出通道数, 一般比整个模块的通道数要多, 类似于倒残差结构
    # ratio: Ghost模块中第一个1 * 1卷积下降通道数的倍数, 一般为2
    # se: 瓶颈模块中是否使用SE注意力机制

    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # ratio = oup / intrinsic
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, padding=dw_size//2, groups=init_channels, bias=False), # groups 分组卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
