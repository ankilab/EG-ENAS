#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

from search_space.config import cfg
from .blocks import (
    activation,
    conv2d,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    init_weights_he,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    pool2d,
    pool2d_cx,
    SE,
)
from torch.nn import Module
import torch.nn as nn
import torch


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "simple_stem_in": SimpleStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
        "res_bottleneck_linear_block": ResBottleneckLinearBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(Module):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, head_width, num_classes, dropout_rate=0.0):
        super(AnyHead, self).__init__()
        self.head_width = head_width
        if head_width > 0:
            self.conv = conv2d(w_in, head_width, 1)
            self.bn = norm2d(head_width)
            self.af = activation()
            w_in = head_width
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def forward(self, x):
        x = self.af(self.bn(self.conv(x))) if self.head_width > 0 else x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, head_width, num_classes):
        if head_width > 0:
            cx = conv2d_cx(cx, w_in, head_width, 1)
            cx = norm2d_cx(cx, head_width)
            w_in = head_width
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx


class VanillaBlock(Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(VanillaBlock, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class BasicTransform(Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(BasicTransform, self).__init__()
        w_se = int(round(w_in * _params["se_r"]))
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_se = int(round(w_in * params["se_r"]))
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBasicBlock(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params,  norm_layer=nn.BatchNorm2d, dilation=(1, 1)):
        super(ResBasicBlock, self).__init__()
        #self.proj, self.bn = None, None
        #if (w_in != w_out) or (stride != 1):
        #    self.proj = conv2d(w_in, w_out, 1, stride=stride)
        #    self.bn = norm2d(w_out)
        self.f = BasicTransform(w_in, w_out, stride, params)
        self.af = activation()

        self.downsample = create_shortcut(
            params["downsample"],
            w_in,
            w_out,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(params["drop_path_rate"]) if params["drop_path_rate"] > 0 else nn.Identity()


    def forward(self, x):
        #x_p = self.bn(self.proj(x)) if self.proj else x
        #return self.af(x_p + self.f(x))
        shortcut=x
        x=self.f(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        return self.af(x)

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BasicTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params, norm_layer=nn.BatchNorm2d, dilation=(1, 1)):
        super(ResBottleneckBlock, self).__init__()
        #self.proj, self.bn = None, None
        #if (w_in != w_out) or (stride != 1):
        #    self.proj = conv2d(w_in, w_out, 1, stride=stride)
        #    self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()
        
        self.downsample = create_shortcut(
            params["downsample"],
            w_in,
            w_out,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(params["drop_path_rate"]) if params["drop_path_rate"] > 0 else nn.Identity()

    def forward(self, x):
        #x_p = self.bn(self.proj(x)) if self.proj else x
        shortcut=x
        x=self.f(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        return self.af(x)

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class ResBottleneckLinearBlock(Module):
    """Residual linear bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckLinearBlock, self).__init__()
        self.has_skip = (w_in == w_out) and (stride == 1)
        self.f = BottleneckTransform(w_in, w_out, stride, params)

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        return BottleneckTransform.complexity(cx, w_in, w_out, stride, params)


class ResStemCifar(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStem(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class SimpleStem(Module):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class AnyStage(Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, params):
        super(AnyStage, self).__init__()
        for i in range(d):
            block = block_fun(w_in, w_out, stride, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, params):
        for _ in range(d):
            cx = block_fun.complexity(cx, w_in, w_out, stride, params)
            stride, w_in = 1, w_out
        return cx


class AnyNet(Module):
    """AnyNet model."""

    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "head_w": cfg.ANYNET.HEAD_W,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,

        }

    def __init__(self, params=None):
        super(AnyNet, self).__init__()
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(p["input_channels"], p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for i, (d, w, s, b, g) in enumerate(zip(*[p[k] for k in keys])):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"], "downsample":p["downsample"], "drop_path_rate":p["drop_path_rate"], "dropout_rate":p["dropout_rate"]}
            print(params)
            stage = AnyStage(prev_w, w, s, d, block_fun, params)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w
        print(p)
        self.head = AnyHead(prev_w, p["head_w"], p["num_classes"], p["dropout_rate"])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        cx = stem_fun.complexity(cx, 3, p["stem_w"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for d, w, s, b, g in zip(*[p[k] for k in keys]):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, params)
            prev_w = w
        cx = AnyHead.complexity(cx, prev_w, p["head_w"], p["num_classes"])
        return cx
    
#############################################################
from timm.layers import ClassifierHead, AvgPool2dSame, ConvNormAct, SEModule, DropPath, GroupNormAct
from timm.layers import create_conv2d

class DropPath(nn.Module):
    """
    Implementation of paper https://arxiv.org/pdf/1603.09382v3.pdf
    From: https://github.com/rwightman/pytorch-image-models/blob/6d8272e92c3d5f13a9fdd91dfe1eb7fae6784589/timm/models/layers/drop.py#L160
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob is None or self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],)+(1,)*(x.ndim-1)
        rand_tensor = keep_prob + torch.rand(shape, dtype=x.dtype,
                                             device=x.device)
        rand_tensor = rand_tensor.floor_()
        out = x.div(keep_prob) * rand_tensor
        return out

def downsample_avg(
        in_chs,
        out_chs,
        kernel_size=1,
        stride=1,
        dilation=1,
        norm_layer=None,
        preact=False,
):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    if preact:
        conv = create_conv2d(in_chs, out_chs, 1, stride=1)
    else:
        conv = ConvNormAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, apply_act=False)
    return nn.Sequential(*[pool, conv])


def downsample_conv(
        in_chs,
        out_chs,
        kernel_size=1,
        stride=1,
        dilation=1,
        norm_layer=None,
        preact=False,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    if preact:
        return create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )
    else:
        return ConvNormAct(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            apply_act=False,
        )


def create_shortcut(
        downsample_type,
        in_chs,
        out_chs,
        kernel_size,
        stride,
        dilation=(1, 1),
        norm_layer=None,
        preact=False,
):
    assert downsample_type in ('avg', 'conv1x1', '', None)
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        dargs = dict(stride=stride, dilation=dilation[0], norm_layer=norm_layer, preact=preact)
        if not downsample_type:
            return None  # no shortcut, no downsample
        elif downsample_type == 'avg':
            return downsample_avg(in_chs, out_chs, **dargs)
        else:
            return downsample_conv(in_chs, out_chs, kernel_size=kernel_size, **dargs)
    else:
        return nn.Identity()  # identity shortcut (no downsample)