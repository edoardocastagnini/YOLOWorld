# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys
import torch.ao.nn.quantized as nnq
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, SeparableConv2d, DepthwiseConv2d, SEBlock
from .transformer import TransformerBlock
from typing import Union, Tuple, Optional, List


__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "PhiNetConvBlock",
    "PhiNet",
    "XiConv",
    "XiNet",
)


def get_xpansion_factor(t_zero, beta, block_id, num_blocks):
    """Compute the expansion factor based on the formula from the paper.

    Arguments
    ---------
    t_zero : float
        The base expansion factor.
    beta : float
        The shape factor.
    block_id : int
        The identifier of the current block.
    num_blocks : int
        The total number of blocks.

    Returns
    -------
    float
        The computed expansion factor.
    """
    return (t_zero * beta) * block_id / num_blocks + t_zero * (
        num_blocks - block_id
    ) / num_blocks




def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    It ensures that all layers have a channel number that is divisible by divisor.

    Arguments
    ---------
    v : int
        The original number of channels.
    divisor : int, optional
        The divisor to ensure divisibility (default is 8).
    min_value : int or None, optional
        The minimum value for the divisible channels (default is None).

    Returns
    -------
    int
        The adjusted number of channels.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(input_shape, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments
    ---------
    input_shape : tuple or list
        Shape of the input tensor (height, width).
    kernel_size : int or tuple
        Size of the convolution kernel.

    Returns
    -------
    tuple
        A tuple representing the zero-padding in the format (left, right, top, bottom).
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_shape[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_shape[0] % 2, 1 - input_shape[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return (
        int(correct[1] - adjust[1]),
        int(correct[1]),
        int(correct[0] - adjust[0]),
        int(correct[0]),
    )


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""
    
    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4): #s is stride and n is number of blocks, e is expansion factor
        """Initializes the ResNetLayer given arguments."""
        # breakpoint()
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
         #x starts [3, 256, 256]
        # breakpoint()
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class XiConv(nn.Module):
    # XiNET convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, compression=4, attention=True, skip_tensor_in=None, skip_channels=1, pool=None, attention_k=3, attention_lite=True, batchnorm=True, dropout_rate=0,  skip_k=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.compression = compression
        self.attention = attention
        self.attention_lite = attention_lite
        self.attention_lite_ch_in = c2//compression
        self.pool = pool
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        if self.compression > 1:
            self.compression_conv = nn.Conv2d(c1, c2//compression, 1, 1,  groups=g, bias=False)
        self.main_conv = nn.Conv2d(c2//compression if compression>1 else c1, c2, k, s,  groups=g, padding=autopad(k, None), bias=False)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
        if attention:
            if attention_lite:
                self.att_pw_conv= nn.Conv2d(c2, self.attention_lite_ch_in, 1, 1, groups=g, bias=False)
            self.att_conv = nn.Conv2d(c2 if not attention_lite else self.attention_lite_ch_in, c2, attention_k, 1, groups=g, padding=autopad(attention_k, None), bias=False)
            self.att_act = nn.Sigmoid()

        if pool:
            self.mp = nn.MaxPool2d(pool)
        if skip_tensor_in:
            breakpoint()
            self.skip_conv = nn.Conv2d(skip_channels, c2//compression, skip_k, 1,  groups=g, padding=autopad(skip_k, None), bias=False)
        if batchnorm:
            self.bn = nn.BatchNorm2d(c2)
        if dropout_rate>0:
            self.do = nn.Dropout(dropout_rate)
    

        


    def forward(self, x):
        s = None
        if isinstance(x, list):
            s = F.adaptive_avg_pool2d(x[1], output_size=x[0].shape[2:])
            s = self.skip_conv(s)
            x = x[0]
        #     print(f'Skip shape {s.shape}')
        # print(f'Input shape {x.shape}')

        # compression convolution
        if self.compression > 1:
            x = self.compression_conv(x)
        if s is not None:
            # print(f'Tensor shape {x.shape}')
            x = x+s
        if self.pool:
            x = self.mp(x)
        # main conv and activation
        x = self.main_conv(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)

        # attention conv
        if self.attention:
            if self.attention_lite:
                att_in=self.att_pw_conv(x)
            else:
                att_in=x
            y = self.att_act(self.att_conv(att_in))
            x = x*y

        
        if self.dropout_rate > 0:
            x = self.do(x)

        # print(f'Output shape {x.shape} \n')
        return x

    #TODO implement fused conv
    def forward_fuse(self, x):
        return self.forward(x)



class XiNet(nn.Module):
    """Defines a XiNet.

    Arguments
    ---------
    input_shape : List
        Shape of the input tensor.
    alpha: float
        Width multiplier.
    gamma : float
        Compression factor.
    num_layers : int = 5
        Number of convolutional blocks.
    num_classes : int
        Number of classes. It is used only when include_top is True.
    include_top : Optional[bool]
        When True, defines an MLP for classification.
    base_filters : int
        Number of base filters for the ConvBlock.
    return_layers : Optional[List]
        Ids of the layers to be returned after processing the foward
        step.

    Example
    -------
    .. doctest::

        >>> from micromind.networks import XiNet
        >>> model = XiNet((3, 224, 224))
    """

    def __init__(
        self,
        input_shape: List,
        alpha: float = 1.0,
        gamma: float = 4.0,
        num_layers: int = 5,
        num_classes=1000,
        include_top=False,
        base_filters: int = 16,
        return_layers: Optional[List] = None,
    ):
        super().__init__()
        breakpoint()


        self._layers = nn.ModuleList([])
        self.input_shape = torch.Tensor(input_shape)
        self.include_top = include_top
        self.return_layers = return_layers
        count_downsample = 0

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_shape[0],
                int(base_filters * alpha),
                7,
                padding=7 // 2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(int(base_filters * alpha)),
            nn.SiLU(),
        )
        count_downsample += 1

        num_filters = [
            int(2 ** (base_filters**0.5 + i)) for i in range(0, num_layers)
        ]
        skip_channels_num = int(base_filters * 2 * alpha)

        for i in range(
            len(num_filters) - 2
        ):  # Account for the last two layers separately
            self._layers.append(
                XiConv(
                    int(num_filters[i] * alpha),
                    int(num_filters[i + 1] * alpha),
                    k=3,
                    s=1,
                    pool=2,
                    skip_tensor_in=(i != 0),
                    skip_res=self.input_shape[1:] / (2**count_downsample),
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )
            count_downsample += 1
            self._layers.append(
                XiConv(
                    int(num_filters[i + 1] * alpha),
                    int(num_filters[i + 1] * alpha),
                    kernel_size=3,
                    stride=1,
                    skip_tensor_in=True,
                    skip_res=self.input_shape[1:] / (2**count_downsample),
                    skip_channels=skip_channels_num,
                    gamma=gamma,
                )
            )

        # Adding the last two layers with attention=False
        self._layers.append(
            XiConv(
                int(num_filters[-2] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=3,
                stride=1,
                skip_tensor_in=True,
                skip_res=self.input_shape[1:] / (2**count_downsample),
                skip_channels=skip_channels_num,
                attention=False,
            )
        )
        # count_downsample += 1
        self._layers.append(
            XiConv(
                int(num_filters[-1] * alpha),
                int(num_filters[-1] * alpha),
                kernel_size=3,
                stride=1,
                skip_tensor_in=True,
                skip_res=self.input_shape[1:] / (2**count_downsample),
                skip_channels=skip_channels_num,
                attention=False,
            )
        )

        if self.return_layers is not None:
            print(f"XiNet configured to return layers {self.return_layers}:")
            for i in self.return_layers:
                print(f"Layer {i} - {self._layers[i].__class__}")

        self.input_shape = input_shape
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(num_filters[-1] * alpha), num_classes),
            )

    def forward(self, x):
        """Computes the forward step of the XiNet.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Output of the network, as defined from
            the init. : Union[torch.Tensor, Tuple]
        """
        x = self.conv1(x)
        skip = None
        ret = []
        for layer_id, layer in enumerate(self._layers):
            if layer_id == 0:
                x = layer(x)
                skip = x
            else:
                x = layer([x, skip])

            if self.return_layers is not None:
                if layer_id in self.return_layers:
                    ret.append(x)

        if self.include_top:
            x = self.classifier(x)

        if self.return_layers is not None:
            return x, ret

        return x



class PhiNetConvBlock(nn.Module):
    """Implements PhiNet's convolutional block.

    Arguments
    ---------
    in_shape : tuple
        Input shape of the conv block.
    expansion : float
        Expansion coefficient for this convolutional block.
    stride: int
        Stride for the conv block.
    filters : int
        Output channels of the convolutional block.
    block_id : int
        ID of the convolutional block.
    has_se : bool
        Whether to include use Squeeze and Excite or not.
    res : bool
        Whether to use the residual connection or not.
    h_swish : bool
        Whether to use HSwish or not.
    k_size : int
        Kernel size for the depthwise convolution.

    """

    def __init__(
        self,
        in_shape,
        expansion,
        stride,
        filters,
        has_se,
        block_id=None,
        res=True,
        h_swish=True,
        k_size=3,
        dp_rate=0.05,
        divisor=1,
    ):
        super(PhiNetConvBlock, self).__init__()

        self.param_count = 0

        self.skip_conn = False

        self._layers = torch.nn.ModuleList()
        in_channels = in_shape[0]

        # Define activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        if block_id:
            # Expand
            conv1 = nn.Conv2d(
                in_channels,
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                kernel_size=1,
                padding=0,
                bias=False,
            )

            bn1 = nn.BatchNorm2d(
                _make_divisible(int(expansion * in_channels), divisor=divisor),
                eps=1e-3,
                momentum=0.999,
            )

            self._layers.append(conv1)
            self._layers.append(bn1)
            self._layers.append(activation)
        
        if stride == 2:
            padding = correct_pad([res, res], 3)

        self._layers.append(nn.Dropout2d(dp_rate))

        d_mul = 1
        in_channels_dw = (
            _make_divisible(int(expansion * in_channels), divisor=divisor)
            if block_id
            else in_channels
        )
        out_channels_dw = in_channels_dw * d_mul
        dw1 = DepthwiseConv2d(
            in_channels=in_channels_dw,
            depth_multiplier=d_mul,
            kernel_size=k_size,
            stride=stride,
            bias=False,
            padding=k_size // 2 if stride == 1 else (padding[1], padding[3]),
        )

        bn_dw1 = nn.BatchNorm2d(
            out_channels_dw,
            eps=1e-3,
            momentum=0.999,
        )

        # It is necessary to reinitialize the activation
        # for functions using Module.children() to work properly.
        # Module.children() does not return repeated layers.
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        self._layers.append(dw1)
        self._layers.append(bn_dw1)
        self._layers.append(activation)

        if has_se:
            num_reduced_filters = _make_divisible(
                max(1, int(out_channels_dw / 6)), divisor=divisor
            )
            se_block = SEBlock(out_channels_dw, num_reduced_filters, h_swish=h_swish)
            self._layers.append(se_block)

        conv2 = nn.Conv2d(
            in_channels=out_channels_dw,
            out_channels=filters,
            kernel_size=1,
            padding=0,
            bias=False,
        )

        bn2 = nn.BatchNorm2d(
            filters,
            eps=1e-3,
            momentum=0.999,
        )

        self._layers.append(conv2)
        self._layers.append(bn2)

        if res and in_channels == filters and stride == 1:
            self.skip_conn = True
            # It serves for the quantization.
            # The behavior remains equivalent for the unquantized models.
            self.op = nnq.FloatFunctional()

    def forward(self, x):
        """Executes the PhiNet convolutional block.

        Arguments
        ---------
        x : torch.Tensor
            Input to the convolutional block.

        Returns
        -------
        torch.Tensor
            Output of the convolutional block.
        """

        if self.skip_conn:
            inp = x

        for layer in self._layers:
            x = layer(x)

        if self.skip_conn:
            return self.op.add(x, inp)  # Equivalent to ``torch.add(a, b)``
        return x


class PhiNet(nn.Module):
    """
    This class implements the PhiNet architecture.

    Arguments
    ---------
    input_shape : tuple
        Input resolution as (C, H, W).
    num_layers : int
        Number of convolutional blocks.
    alpha: float
        Width multiplier for PhiNet architecture.
    beta : float
        Shape factor of PhiNet.
    t_zero : float
        Base expansion factor for PhiNet.
    include_top : bool
        Whether to include classification head or not.
    num_classes : int
        Number of classes for the classification head.
    compatibility : bool
        `True` to maximise compatibility among embedded platforms (changes network).

    """

    def get_complexity(self):
        """Returns MAC and number of parameters of initialized architecture.

        Returns
        -------
            Dictionary with complexity characterization of the network. : dict

        Example
        -------
        .. doctest::

            >>> from micromind.networks import PhiNet
            >>> model = PhiNet((3, 224, 224))
            >>> model.get_complexity()
            {'MAC': 9817670, 'params': 30917}
        """
        temp = summary(
            self, input_data=torch.zeros([1] + list(self.input_shape)), verbose=0
        )

        return {"MAC": temp.total_mult_adds, "params": temp.total_params}

    def get_MAC(self):
        """Returns number of MACs for this architecture.

        Returns
        -------
            Number of MAC for this network. : int

        Example
        -------
        .. doctest::

            >>> from micromind.networks import PhiNet
            >>> model = PhiNet((3, 224, 224))
            >>> model.get_MAC()
            9817670
        """
        return self.get_complexity()["MAC"]

    def get_params(self):
        """Returns number of params for this architecture.

        Returns
        -------
            Number of parameters for this network. : int

        Example
        -------
        .. doctest::

            >>> from micromind.networks import PhiNet
            >>> model = PhiNet((3, 224, 224))
            >>> model.get_params()
            30917
        """
        return self.get_complexity()["params"]

    def __init__(
        self,
        input_shape: List[int],
        num_layers: int = 7,  # num_layers
        alpha: float = 0.2,
        beta: float = 1.0,
        t_zero: float = 6,
        include_top: bool = False,
        num_classes: int = 10,
        compatibility: bool = False,
        downsampling_layers: List[int] = [5, 7],  # S2
        conv5_percent: float = 0.0,  # S2
        first_conv_stride: int = 2,  # S2
        residuals: bool = True,  # S2
        conv2d_input: bool = False,  # S2
        pool: bool = False,  # S2
        h_swish: bool = True,  # S1
        squeeze_excite: bool = True,  # S1
        divisor: int = 1,
        return_layers=None,
        first_conv_filters: int = 0,
        b1_filters: int = 0,
        b2_filters: int = 0,
    ) -> None:
        super(PhiNet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.t_zero = t_zero
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.return_layers = return_layers
        self.first_conv_filters = first_conv_filters
        self.b1_filters = b1_filters
        self.b2_filters = b2_filters
        #good
        if compatibility:  # disables operations hard for some platforms
            h_swish = False
            squeeze_excite = False

        if not isinstance(num_layers, int):
            num_layers = round(num_layers)

        assert len(input_shape) == 3, "Expected 3 elements list as input_shape."
        in_channels = input_shape[0]
        res = max(input_shape[1], input_shape[2])  # assumes squared input
        self.input_shape = input_shape
        self.classify = include_top
        self._layers = torch.nn.ModuleList()

        # Define self.activation function
        if h_swish:
            activation = nn.Hardswish(inplace=True)
        else:
            activation = ReLUMax(6)

        mp = nn.MaxPool2d((2, 2))

        if not conv2d_input:
            pad = nn.ZeroPad2d(
                padding=correct_pad([res, res], 3),
            )

            self._layers.append(pad)

            sep1 = SeparableConv2d(
                in_channels,
                _make_divisible(int(first_conv_filters * alpha), divisor=divisor),
                kernel_size=3,
                stride=(first_conv_stride, first_conv_stride),
                padding=0,
                bias=False,
                activation=activation,
            )

            self._layers.append(sep1)
            # self._layers.append(activation)
            block1 = PhiNetConvBlock(
                in_shape=(
                    _make_divisible(int(first_conv_filters * alpha), divisor=divisor),
                    res / first_conv_stride,
                    res / first_conv_stride,
                ),
                filters=_make_divisible(int(b1_filters * alpha), divisor=divisor),
                stride=1,
                expansion=1,
                has_se=False,
                res=residuals,
                h_swish=h_swish,
                divisor=divisor,
            )

            self._layers.append(block1)
        else:
            c1 = nn.Conv2d(
                in_channels, int(b1_filters * alpha), kernel_size=(3, 3), bias=False
            )

            bn_c1 = nn.BatchNorm2d(int(b1_filters * alpha))

            self._layers.append(c1)
            self._layers.append(activation)
            self._layers.append(bn_c1)

        block2 = PhiNetConvBlock(
            (
                _make_divisible(int(b1_filters * alpha), divisor=divisor),
                res / first_conv_stride,
                res / first_conv_stride,
            ),
            filters=_make_divisible(int(b1_filters * alpha), divisor=divisor),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 1, num_layers),
            block_id=1,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            divisor=divisor,
        )

        block3 = PhiNetConvBlock(
            (
                _make_divisible(int(b1_filters * alpha), divisor=divisor),
                res / first_conv_stride / 2,
                res / first_conv_stride / 2,
            ),
            filters=_make_divisible(int(b1_filters * alpha), divisor=divisor),
            stride=1,
            expansion=get_xpansion_factor(t_zero, beta, 2, num_layers),
            block_id=2,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            divisor=divisor,
        )

        block4 = PhiNetConvBlock(
            (
                _make_divisible(int(b1_filters * alpha), divisor=divisor),
                res / first_conv_stride / 2,
                res / first_conv_stride / 2,
            ),
            filters=_make_divisible(int(b2_filters * alpha), divisor=divisor),
            stride=2 if (not pool) else 1,
            expansion=get_xpansion_factor(t_zero, beta, 3, num_layers),
            block_id=3,
            has_se=squeeze_excite,
            res=residuals,
            h_swish=h_swish,
            divisor=divisor,
        )
        self._layers.append(block2)
        if pool:
            self._layers.append(mp)
        self._layers.append(block3)
        self._layers.append(block4)
        if pool:
            self._layers.append(mp)

        block_id = 4
        block_filters = b2_filters
        spatial_res = res / first_conv_stride / 4
        in_channels_next = _make_divisible(int(b2_filters * alpha), divisor=divisor)
        while num_layers >= block_id:
            if block_id in downsampling_layers:
                block_filters *= 2
                if pool:
                    self._layers.append(mp)
            pn_block = PhiNetConvBlock(
                (in_channels_next, spatial_res, spatial_res),
                filters=_make_divisible(int(block_filters * alpha), divisor=divisor),
                stride=(2 if (block_id in downsampling_layers) and (not pool) else 1),
                expansion=get_xpansion_factor(t_zero, beta, block_id, num_layers),
                block_id=block_id,
                has_se=squeeze_excite,
                res=residuals,
                h_swish=h_swish,
                k_size=(5 if (block_id / num_layers) > (1 - conv5_percent) else 3),
                divisor=divisor,
            )
            self._layers.append(pn_block)
            in_channels_next = _make_divisible(
                int(block_filters * alpha), divisor=divisor
            )
            spatial_res = (
                spatial_res / 2 if block_id in downsampling_layers else spatial_res
            )
            block_id += 1

        if include_top:
            # Includes classification head if required
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(
                    _make_divisible(int(block_filters * alpha), divisor=divisor),
                    num_classes,
                    bias=True,
                ),
            )

        if self.return_layers is not None:
            print(f"PhiNet configured to return layers {self.return_layers}:")
            for i in self.return_layers:
                print(f"Layer {i} - {self._layers[i].__class__}")
                
    def forward(self, x):
     #x starts [3, 256, 256]
        """Executes PhiNet network

        Arguments
        -------
        x : torch.Tensor
            Network input.

        Returns
        ------
            Logits if `include_top=True`, otherwise embeddings : torch.Tensor
        """
        ret = []
        for i, layers in enumerate(self._layers):
            x = layers(x)
            if self.return_layers is not None:
                if i in self.return_layers:
                    ret.append(x)
        if self.classify:
            x = self.classifier(x)

        if self.return_layers is not None:
            #if ret is a list of 1 element, return the element
            if len(ret) == 1:
                return ret[0]
            return ret #x ends as a list [128,32,32], [256,16,16], [512,8,8]
        return x
    

                        
                        

# Crea una funzione per salvare l'output in un file
def save_model_summary_to_file(model, input_size, file_name):
    # Apri il file in modalità scrittura
    with open(file_name, 'w') as f:
        # Salva l'output originale di sys.stdout
        original_stdout = sys.stdout
        try:
            # Reindirizza sys.stdout al file
            sys.stdout = f
            # Stampa il sommario del modello nel file
            summary(model, input_size, device='cuda')  

            torchinfo.summary(model) 
            
            print_model_layers(model)
        finally:
            # Ripristina l'output originale
            sys.stdout = original_stdout



def print_model_layers(module, indent=0):
    """
    Ricorsivamente attraversa i moduli e stampa i layer dentro i container come ModuleList o Sequential,
    scendendo in profondità per esplorare tutti i moduli.
    """
    for name, layer in module.named_children():
        print(" " * indent + f"{name} - {layer.__class__.__name__}")
        
        
        # Se il layer è un container come ModuleList, Sequential, o un altro nn.Module
        if isinstance(layer, (nn.ModuleList, nn.Sequential, nn.Module)):
            # Se ha ulteriori moduli all'interno, esegui la funzione ricorsivamente
            print_model_layers(layer, indent + 4)

