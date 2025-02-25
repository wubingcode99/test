import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.MLCA import MLCA
from ultralytics.nn.modules.EMA import EMA
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class C2f_mlca(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5,n_div=4, pconv_fw_type='split_cat'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels计算隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 创建一个卷积cv1
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)创建一个卷积cv2
        self.m = nn.ModuleList(Bottleneck_mlca(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck_mlca(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, n_div=4, pconv_fw_type='split_cat'):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        #self.cv1 = Conv(c1, c_, k[0], 1)
        #self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        self.cv1 = Partial_conv3(
            c1,
            n_div,
            pconv_fw_type
        )
        self.cv2 = Partial_conv3(
            c2,
            n_div,
            pconv_fw_type
        )
        self.attention = MLCA(c2)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        #return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))
