import os
import torch
from torch.nn import functional as F

import torch
import torch.nn as nn
from util import pyutils
from module.dataloader import get_dataloader
from module.model import get_model
from module.optimizer import get_optimizer, get_scheduler

def train(args, validate=True):
    # torch.autograd.set_detect_anomaly(True)
    print('==========training start!============')
    
    ## data, optimizer, model setting
    train_loader = get_dataloader(args) # load dataset
    temp = args.datalist
    args.datalist = args.vallist
    val_loader = get_dataloader(args) # load validation dataset
    args.datalist = temp
    
    args.max_iters = args.epochs * len(train_loader) ## max_iter adjust before optimizer definition

    if 'vit' in args.network_type:
        model = get_model(args).to(args.device) # .to(rank) # load network and its pre-trained model
        model.train()
        optimizer = get_optimizer(args, model, args.max_iters) # set optimizer
        scheduler = get_scheduler(args, optimizer)
    else:
        model = get_model(args).to(args.device) # .to(rank) # load network and its pre-trained model
        model.train()
        model = nn.DataParallel(model).to(args.device)
        model.train()
        optimizer = get_optimizer(args, model.module, args.max_iters) # set optimizer
        scheduler = get_scheduler(args, optimizer)
    
    
    ## printing utility
    avg_meter = pyutils.AverageMeter('loss')
    timer = pyutils.Timer("Session started: ")
    
    for ep in range(args.epochs):
        for iteration, pack in enumerate(train_loader):
            img = pack['img'].to(args.device)
            label = pack['label'].to(args.device)

            # import pdb; pdb.set_trace()
            pred = model(img) # does not kill the cuda
            loss = F.multilabel_soft_margin_loss(pred, label)

            # loss & backward pass
            optimizer.zero_grad()
            loss.backward() # does not kill the cuda
            
            if 'vit' in args.network_type:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step() # does not kill the cuda
            
            avg_meter.add({'loss': loss.item()})
            if iteration % 100 == 5: # optimizer.global_step-1
                timer.update_progress((ep * len(train_loader) + iteration + 1) / args.max_iters)

                print('Iter:%5d/%5d' % (ep * len(train_loader) + iteration, args.max_iters),
                    'Loss:%.4f' % (avg_meter.pop('loss')),
                    'Rem:%s' % (timer.get_est_remain()),
                    'lr: %.4f' % (optimizer.param_groups[0]['lr']), 
                    flush=True)
                
                if validate == True:
                    validation(model, val_loader, args)
                
            timer.reset_stage()
        if scheduler is not None: scheduler.step()
        
    if args.network_type == 'resnet50':
        torch.save(model.module.state_dict(), os.path.join(args.weight_root, 'resnet50_cam.pth'))
    elif args.network_type == 'vit':
        torch.save(model.state_dict(), os.path.join(args.weight_root, 'vit_cam.pth'))
    elif args.network_type == 'resnet50_coco':
        torch.save(model.module.state_dict(), os.path.join(args.weight_root, 'resnet50_cam_coco.pth'))
    elif args.network_type == 'vit_coco':
        torch.save(model.state_dict(), os.path.join(args.weight_root, 'vit_cam_coco.pth'))
    else:
        raise NotImplementedError('No models like that!')
    
    
def run_train(args):
    train(args, False)
    torch.cuda.empty_cache()
    
    
def validation(model, val_loader, args):
    ## printing utility
    avg_meter = pyutils.AverageMeter('loss')
    
    # with torch.no_grad(): # cannot use torch.no_grad because gradient hooking in ViT
    for iteration, pack in enumerate(val_loader):
        img = pack['img'].to(args.device)
        label = pack['label'].to(args.device)
        
        pred = model(img)
        loss = F.multilabel_soft_margin_loss(pred, label)

        avg_meter.add({'loss': loss.item()})
    print('Validation Loss:%.4f' % (avg_meter.pop('loss')), flush=True)
    torch.cuda.empty_cache()