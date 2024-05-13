"""RegNet X, Y, Z, and more

Paper: `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py

Paper: `Fast and Accurate Model Scaling` - https://arxiv.org/abs/2103.06877
Original Impl: None

Based on original PyTorch impl linked above, but re-wrote to use my own blocks (adapted from ResNet here)
and cleaned up with more descriptive variable names.

Weights from original pycls impl have been modified:
* first layer from BGR -> RGB as most PyTorch models are
* removed training specific dict entries from checkpoints and keep model state_dict only
* remap names to match the ones here

Supports weight loading from torchvision and classy-vision (incl VISSL SEER)

A number of custom timm model definitions additions including:
* stochastic depth, gradient checkpointing, layer-decay, configurable dilation
* a pre-activation 'V' variant
* only known RegNet-Z model definitions with pretrained weights

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from timm.layers import ClassifierHead, AvgPool2dSame, ConvNormAct, SEModule, DropPath, GroupNormAct
from timm.layers import get_act_layer, get_norm_act_layer, create_conv2d, make_divisible
#from ._features import feature_take_indices
#from ._manipulate import checkpoint_seq, named_apply

__all__ = ['RegNet', 'RegNetCfg']  # model_registry will add each entrypoint fn to this

def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x

def feature_take_indices(
        num_blocks: int,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
) -> Tuple[List[int], int]:
    if indices is None:
        indices = num_blocks  # all blocks if None
    if torch.jit.is_scripting():
        return _take_indices_jit(num_blocks, indices)
    else:
        # NOTE non-jit returns Set[int] instead of List[int] but torchscript can't handle that anno
        return _take_indices(num_blocks, indices)
#################################


@dataclass
class RegNetCfg:
    depth: int = 21
    w0: int = 80
    wa: float = 42.63
    wm: float = 2.66
    group_size: int = 24
    bottle_ratio: float = 1.
    se_ratio: float = 0.
    group_min_ratio: float = 0.
    stem_width: int = 32
    downsample: Optional[str] = 'conv1x1'
    linear_out: bool = False
    preact: bool = False
    num_features: int = 0
    act_layer: Union[str, Callable] = 'relu'
    norm_layer: Union[str, Callable] = 'batchnorm'


def quantize_float(f, q):
    """Converts a float to the closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups, min_ratio=0.):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    if min_ratio:
        # torchvision uses a different rounding scheme for ensuring bottleneck widths divisible by group widths
        bottleneck_widths = [make_divisible(w_bot, g, min_ratio) for w_bot, g in zip(bottleneck_widths, groups)]
    else:
        bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, group_size, quant=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % quant == 0
    # TODO dWr scaling?
    # depth = int(depth * (scale ** 0.1))
    # width_scale = scale ** 0.4  # dWr scale, exp 0.8 / 2, applied to both group and layer widths
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = np.round(np.divide(width_initial * np.power(width_mult, width_exps), quant)) * quant
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    groups = np.array([group_size for _ in range(num_stages)])
    return widths.astype(int).tolist(), num_stages, groups.astype(int).tolist()


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


class Bottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            dilation=(1, 1),
            bottle_ratio=1,
            group_size=1,
            se_ratio=0.25,
            downsample='conv1x1',
            linear_out=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_block=None,
            drop_path_rate=0.,
    ):
        super(Bottleneck, self).__init__()
        act_layer = get_act_layer(act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvNormAct(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
            drop_layer=drop_block,
            **cargs,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.conv3 = ConvNormAct(bottleneck_chs, out_chs, kernel_size=1, apply_act=False, **cargs)
        self.act3 = nn.Identity() if linear_out else act_layer()
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        x = self.act3(x)
        return x


class PreBottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            dilation=(1, 1),
            bottle_ratio=1,
            group_size=1,
            se_ratio=0.25,
            downsample='conv1x1',
            linear_out=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_block=None,
            drop_path_rate=0.,
    ):
        super(PreBottleneck, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        self.norm1 = norm_act_layer(in_chs)
        self.conv1 = create_conv2d(in_chs, bottleneck_chs, kernel_size=1)
        self.norm2 = norm_act_layer(bottleneck_chs)
        self.conv2 = create_conv2d(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.norm3 = norm_act_layer(bottleneck_chs)
        self.conv3 = create_conv2d(bottleneck_chs, out_chs, kernel_size=1)
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            preact=True,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        pass

    def forward(self, x):
        x = self.norm1(x)
        shortcut = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.norm3(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        return x


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(
            self,
            depth,
            in_chs,
            out_chs,
            stride,
            dilation,
            drop_path_rates=None,
            block_fn=Bottleneck,
            **block_kwargs,
    ):
        super(RegStage, self).__init__()
        self.grad_checkpointing = False

        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = (first_dilation, dilation)
            dpr = drop_path_rates[i] if drop_path_rates is not None else 0.
            name = "b{}".format(i + 1)
            self.add_module(
                name,
                block_fn(
                    block_in_chs,
                    out_chs,
                    stride=block_stride,
                    dilation=block_dilation,
                    drop_path_rate=dpr,
                    **block_kwargs,
                )
            )
            first_dilation = dilation

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.children(), x)
        else:
            for block in self.children():
                x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet-X, Y, and Z Models

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(
            self,
            cfg: RegNetCfg,
            in_chans=3,
            num_classes=1000,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.,
            drop_path_rate=0.,
            zero_init_last=True,
            **kwargs,
    ):
        """

        Args:
            cfg (RegNetCfg): Model architecture configuration
            in_chans (int): Number of input channels (default: 3)
            num_classes (int): Number of classifier classes (default: 1000)
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            global_pool (str): Global pooling type (default: 'avg')
            drop_rate (float): Dropout rate (default: 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default: 0.)
            zero_init_last (bool): Zero-init last weight of residual path
            kwargs (dict): Extra kwargs overlayed onto cfg
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        cfg = replace(cfg, **kwargs)  # update cfg with extra passed kwargs

        # Construct the stem
        stem_width = cfg.stem_width
        na_args = dict(act_layer=cfg.act_layer, norm_layer=cfg.norm_layer)
        if cfg.preact:
            self.stem = create_conv2d(in_chans, stem_width, 3, stride=2)
        else:
            self.stem = ConvNormAct(in_chans, stem_width, 3, stride=2, **na_args)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 2
        per_stage_args, common_args = self._get_stage_args(
            cfg,
            output_stride=output_stride,
            drop_path_rate=drop_path_rate,
        )
        #assert len(per_stage_args) == 4
        self.num_stages=len(per_stage_ars)
        block_fn = PreBottleneck if cfg.preact else Bottleneck
        for i, stage_args in enumerate(per_stage_args):
            stage_name = "s{}".format(i + 1)
            self.add_module(
                stage_name,
                RegStage(
                    in_chs=prev_width,
                    block_fn=block_fn,
                    **stage_args,
                    **common_args,
                )
            )
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        if cfg.num_features:
            self.final_conv = ConvNormAct(prev_width, cfg.num_features, kernel_size=1, **na_args)
            self.num_features = cfg.num_features
        else:
            final_act = cfg.linear_out or cfg.preact
            self.final_conv = get_act_layer(cfg.act_layer)() if final_act else nn.Identity()
            self.num_features = prev_width
        self.head = ClassifierHead(
            in_features=self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def _get_stage_args(self, cfg: RegNetCfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        widths, num_stages, stage_gs = generate_regnet(cfg.wa, cfg.w0, cfg.wm, cfg.depth, cfg.group_size)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_br = [cfg.bottle_ratio for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, sum(stage_depths)), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_gs = adjust_widths_groups_comp(
            stage_widths, stage_br, stage_gs, min_ratio=cfg.group_min_ratio)
        arg_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_size', 'drop_path_rates']
        per_stage_args = [
            dict(zip(arg_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_br, stage_gs, stage_dpr)
        ]
        common_args = dict(
            downsample=cfg.downsample,
            se_ratio=cfg.se_ratio,
            linear_out=cfg.linear_out,
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
        )
        print(per_stage_args)
        return per_stage_args, common_args

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^s(\d+)' if coarse else r'^s(\d+)\.b(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in list(self.children())[1:-1]:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int], Tuple[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        intermediates = []
        take_indices, max_index = feature_take_indices(5, indices)

        # forward pass
        feat_idx = 0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        layer_names = ('s1', 's2', 's3', 's4')
        if stop_early:
            layer_names = layer_names[:max_index]
        for n in layer_names:
            feat_idx += 1
            x = getattr(self, n)(x)  # won't work with torchscript, but keeps code reasonable, FML
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        if feat_idx == 4:
            x = self.final_conv(x)

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int], Tuple[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(5, indices)
        layer_names = ('s1', 's2', 's3', 's4')
        layer_names = layer_names[max_index:]
        for n in layer_names:
            setattr(self, n, nn.Identity())
        if max_index < 4:
            self.final_conv = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        for i in range(1, self.num_stages+1):
            x=getattr(self, f's{i}')(x)
        #x = self.s1(x)
        #x = self.s2(x)
        #x = self.s3(x)
        #x = self.s4(x)
        #x = self.s5(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name='', zero_init_last=False):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()




# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    # RegNet-X
    regnetx_002=RegNetCfg(w0=24, wa=36.44, wm=2.49, group_size=8, depth=13),
    regnetx_004=RegNetCfg(w0=24, wa=24.48, wm=2.54, group_size=16, depth=22),
    regnetx_004_tv=RegNetCfg(w0=24, wa=24.48, wm=2.54, group_size=16, depth=22, group_min_ratio=0.9),
    regnetx_006=RegNetCfg(w0=48, wa=36.97, wm=2.24, group_size=24, depth=16),
    regnetx_008=RegNetCfg(w0=56, wa=35.73, wm=2.28, group_size=16, depth=16),
    regnetx_016=RegNetCfg(w0=80, wa=34.01, wm=2.25, group_size=24, depth=18),
    regnetx_032=RegNetCfg(w0=88, wa=26.31, wm=2.25, group_size=48, depth=25),
    regnetx_040=RegNetCfg(w0=96, wa=38.65, wm=2.43, group_size=40, depth=23),
    regnetx_064=RegNetCfg(w0=184, wa=60.83, wm=2.07, group_size=56, depth=17),
    regnetx_080=RegNetCfg(w0=80, wa=49.56, wm=2.88, group_size=120, depth=23),
    regnetx_120=RegNetCfg(w0=168, wa=73.36, wm=2.37, group_size=112, depth=19),
    regnetx_160=RegNetCfg(w0=216, wa=55.59, wm=2.1, group_size=128, depth=22),
    regnetx_320=RegNetCfg(w0=320, wa=69.86, wm=2.0, group_size=168, depth=23),

    # RegNet-Y
    regnety_002=RegNetCfg(w0=24, wa=36.44, wm=2.49, group_size=8, depth=13, se_ratio=0.25),
    regnety_004=RegNetCfg(w0=48, wa=27.89, wm=2.09, group_size=8, depth=16, se_ratio=0.25),
    regnety_006=RegNetCfg(w0=48, wa=32.54, wm=2.32, group_size=16, depth=15, se_ratio=0.25),
    regnety_008=RegNetCfg(w0=56, wa=38.84, wm=2.4, group_size=16, depth=14, se_ratio=0.25),
    regnety_008_tv=RegNetCfg(w0=56, wa=38.84, wm=2.4, group_size=16, depth=14, se_ratio=0.25, group_min_ratio=0.9),
    regnety_016=RegNetCfg(w0=48, wa=20.71, wm=2.65, group_size=24, depth=27, se_ratio=0.25),
    regnety_032=RegNetCfg(w0=80, wa=42.63, wm=2.66, group_size=24, depth=21, se_ratio=0.25),
    regnety_040=RegNetCfg(w0=96, wa=31.41, wm=2.24, group_size=64, depth=22, se_ratio=0.25),
    regnety_064=RegNetCfg(w0=112, wa=33.22, wm=2.27, group_size=72, depth=25, se_ratio=0.25),
    regnety_080=RegNetCfg(w0=192, wa=76.82, wm=2.19, group_size=56, depth=17, se_ratio=0.25),
    regnety_080_tv=RegNetCfg(w0=192, wa=76.82, wm=2.19, group_size=56, depth=17, se_ratio=0.25, group_min_ratio=0.9),
    regnety_120=RegNetCfg(w0=168, wa=73.36, wm=2.37, group_size=112, depth=19, se_ratio=0.25),
    regnety_160=RegNetCfg(w0=200, wa=106.23, wm=2.48, group_size=112, depth=18, se_ratio=0.25),
    regnety_320=RegNetCfg(w0=232, wa=115.89, wm=2.53, group_size=232, depth=20, se_ratio=0.25),
    regnety_640=RegNetCfg(w0=352, wa=147.48, wm=2.4, group_size=328, depth=20, se_ratio=0.25),
    regnety_1280=RegNetCfg(w0=456, wa=160.83, wm=2.52, group_size=264, depth=27, se_ratio=0.25),
    regnety_2560=RegNetCfg(w0=640, wa=230.83, wm=2.53, group_size=373, depth=27, se_ratio=0.25),
    #regnety_2560=RegNetCfg(w0=640, wa=124.47, wm=2.04, group_size=848, depth=27, se_ratio=0.25),

    # Experimental
    regnety_040_sgn=RegNetCfg(
        w0=96, wa=31.41, wm=2.24, group_size=64, depth=22, se_ratio=0.25,
        act_layer='silu', norm_layer=partial(GroupNormAct, group_size=16)),

    # regnetv = 'preact regnet y'
    regnetv_040=RegNetCfg(
        depth=22, w0=96, wa=31.41, wm=2.24, group_size=64, se_ratio=0.25, preact=True, act_layer='silu'),
    regnetv_064=RegNetCfg(
        depth=25, w0=112, wa=33.22, wm=2.27, group_size=72, se_ratio=0.25, preact=True, act_layer='silu',
        downsample='avg'),

    # RegNet-Z (unverified)
    regnetz_005=RegNetCfg(
        depth=21, w0=16, wa=10.7, wm=2.51, group_size=4, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=1024, act_layer='silu',
    ),
    regnetz_040=RegNetCfg(
        depth=28, w0=48, wa=14.5, wm=2.226, group_size=8, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=0, act_layer='silu',
    ),
    regnetz_040_h=RegNetCfg(
        depth=28, w0=48, wa=14.5, wm=2.226, group_size=8, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=1536, act_layer='silu',
    ),
)


