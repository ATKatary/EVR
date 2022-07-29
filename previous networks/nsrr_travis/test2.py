import sys
import os
import math

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision

import nvidia
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

sys.path.append(os.path.abspath('flownet2'))
sys.path.append(os.path.abspath('flownet2/utils'))
from models import FlowNet2, FlowNet2SD
import flow_utils

import torchvision.models as models
import pytorch_msssim 

import time

from utils import flownet_test

# allow maximum memory per GPU
#torch.cuda.set_per_process_memory_fraction(1.0, 1)
#torch.cuda.set_per_process_memory_fraction(1.0, 0)

flownet_test()
