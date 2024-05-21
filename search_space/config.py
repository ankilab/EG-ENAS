#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import os

#from pycls.core.io import pathmgr
from yacs.config import CfgNode
from iopath.common.file_io import PathManagerFactory


# instantiate global path manager for pycls
pathmgr = PathManagerFactory.get()

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ---------------------------------- Model options ----------------------------------- #
_C.MODEL = CfgNode()

# Model type
_C.MODEL.TYPE = ""



# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Activation function (relu or silu/swish)
_C.MODEL.ACTIVATION_FUN = "relu"

# Perform activation inplace if implemented
_C.MODEL.ACTIVATION_INPLACE = True

# Model scaling parameters, see models/scaler.py (has no effect unless scaler is used)
_C.MODEL.SCALING_TYPE = ""
_C.MODEL.SCALING_FACTOR = 1.0

# ---------------------------------- RegNet options ---------------------------------- #
_C.REGNET = CfgNode()

# Stem type
_C.REGNET.STEM_TYPE = "res_stem_cifar"

# Stem width
_C.REGNET.STEM_W = 32

# Block type
_C.REGNET.BLOCK_TYPE = "res_bottleneck_block"

# Stride of each stage
_C.REGNET.STRIDE = 2

# Squeeze-and-Excitation (RegNetY)
_C.REGNET.SE_ON = False
_C.REGNET.SE_R = 0.25

# Depth
_C.REGNET.DEPTH = 10

# Initial width
_C.REGNET.W0 = 32

# Slope
_C.REGNET.WA = 5.0

# Quantization
_C.REGNET.WM = 2.5

# Group width
_C.REGNET.GROUP_W = 16

# Bottleneck multiplier (bm = 1 / b from the paper)
_C.REGNET.BOT_MUL = 1.0

# Head width for first conv in head (if 0 conv is omitted, as is the default)
_C.REGNET.HEAD_W = 0

#Downsample for the residual connection. "avg" as in resnet-D, "conv1x1" as in Regnet. 
_C.REGNET.DOWNSAMPLE = "avg"
#Stochastic drop path rate. Works as regularization
_C.REGNET.DROP_RATE=0.0
#Dropout in last layer
_C.REGNET.DROPOUT=0.0



# -------------------------------- Batch norm options -------------------------------- #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False




# -------------------------------- Layer norm options -------------------------------- #
_C.LN = CfgNode()

# LN epsilon
_C.LN.EPS = 1e-5



# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


# --------------------------------- Deprecated keys ---------------------------------- #
_C.register_deprecated_key("MEM")
_C.register_deprecated_key("MEM.RELU_INPLACE")
_C.register_deprecated_key("OPTIM.GAMMA")
_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")
_C.register_deprecated_key("PORT")
_C.register_deprecated_key("TRAIN.EVAL_PERIOD")
_C.register_deprecated_key("TRAIN.CHECKPOINT_PERIOD")



def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)