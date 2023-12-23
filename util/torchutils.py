import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
import math


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def split_dataset(dataset, n_splits):
    
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


class StepOptimizer(torch.optim.SGD):
    
    def __init__(self, params, lr, weight_decay, max_step, step_list = [5,10,15], momentum=0.9):
        super().__init__(params, lr, weight_decay, nesterov=True)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        self.step_list = step_list
        self.lr_mult = 1

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        
        if self.global_step in self.step_list:
            self.lr_mult = self.lr_mult * 0.1

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * self.lr_mult

        super().step(closure)

        self.global_step += 1
        
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

        
class MaskingCELoss(nn.Module):
    def __init__(self, ignore_index=255, mask_val=1e-5):
        super(MaskingCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.mask_val = mask_val
        
    def forward(self, inputs, targets):
        device = inputs.device
        b,c,h,w = inputs.shape
                
        inputs = inputs.permute(0,2,3,1).reshape(-1, c)
        targets = targets.reshape(-1)
        rows = torch.arange(0,len(targets)).to(device, non_blocking=True)
        logs = F.log_softmax(inputs, dim=-1)
        
        # clearing ignore target
        logs_mask = torch.ones_like(logs)
        logs_mask[targets==self.ignore_index, :] = 0
        targets_mask = torch.ones_like(targets)
        targets_mask[targets==self.ignore_index] = 0
                
        # clearing ignore target
        logs = logs * logs_mask
        targets = targets * targets_mask

        # getting log likelihood
        out = logs[rows, targets]
        
        mask = torch.zeros_like(out)
        mask[targets!=0] = self.mask_val
        
        out = out * mask
        
        return -out.sum()/len(out)