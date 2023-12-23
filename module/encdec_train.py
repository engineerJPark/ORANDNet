import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from util import pyutils
from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer
from metadata.dataset import load_img_id_list, load_img_label_list_from_npy

torch.backends.cudnn.enabled = True

def train(args):
    print('==========training start!============')

    model = get_model(args) # .to(rank) # load network and its pre-trained model
    dp_model = nn.DataParallel(model).cuda()
    dp_model.module.train()
    
    train_loader = get_dataloader(args) # load dataset
    args.max_iters = 5000
    optimizer = get_optimizer(args, dp_model.module, args.max_iters) # set optimizer

    ## printing utility
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")

    # for ep in range(args.epochs):

    for iteration, pack in enumerate(train_loader):
        input_cam_highres = pack['input_cam_highres'].cuda(non_blocking=True) # (B, C, H, W)
        label_cam_highres = pack['label_cam_highres'].cuda(non_blocking=True) # (B, C, H, W)
        
        pred_cam_highres = list()
        for idx in range(input_cam_highres.shape[1]):
            pred_cam_highres.append(dp_model(input_cam_highres[:, idx].unsqueeze(dim=0)).squeeze(dim=0)) # output is (B, H, W)
        pred_cam_highres = torch.cat(pred_cam_highres, dim=0).unsqueeze(dim=0) # (B, C, H, W)
        
        # loss & backward pass
        optimizer.zero_grad()
        loss = F.cross_entropy(pred_cam_highres, label_cam_highres)
        loss.backward()
        optimizer.step()

        avg_meter.add({'loss': loss.item()})
        if (optimizer.global_step-1) % 200 == 0:
            timer.update_progress(optimizer.global_step / args.max_iters)

            print('Iter:%5d/%5d' % (iteration, args.max_iters),
                'Loss:%.4f' % (avg_meter.pop('loss')),
                'Rem:%s' % (timer.get_est_remain()),
                'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)
        timer.reset_stage()

        if iteration > args.max_iters:
            break

    if args.network_type == 'encdec':
        torch.save(dp_model.module.state_dict(), os.path.join(args.weight_root, 'encdec.pth'))
    elif args.network_type == 'encdec_amnmct':
        torch.save(dp_model.module.state_dict(), os.path.join(args.weight_root, 'encdec_amnmct.pth'))
    elif args.network_type == 'encdec_coco':
        torch.save(dp_model.module.state_dict(), os.path.join(args.weight_root, 'encdec_coco.pth'))
    elif args.network_type == 'encdec_amnmct_coco':
        torch.save(dp_model.module.state_dict(), os.path.join(args.weight_root, 'encdec_amnmct_coco.pth'))
    else:
        raise NotImplementedError('No models like that!')

def run_encdec(args):
    train(args)
    torch.cuda.empty_cache()