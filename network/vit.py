import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

from lrp.baselines.ViT.ViT_LRP import deit_base_patch16_512 as vit_LRP
from lrp.baselines.ViT.ViT_LRP import deit_base_patch16_512_adl as vit_LRP_adl
from lrp.baselines.ViT.ViT_explanation_generator import LRP

def vit(args):
    model = vit_LRP(pretrained=True, num_classes=args.num_classes)
    return model