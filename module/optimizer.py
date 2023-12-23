import torch
from util import torchutils

def get_optimizer(args, model, max_step=None):
    if ('resnet50' in args.network_type) or (args.network_type=='vgg'): 
        param_groups = model.trainable_parameters()
        optimizer = torchutils.PolyOptimizer([
            {'params': param_groups[0], 'lr': args.res_lr, 'weight_decay': args.res_wt_dec},
            {'params': param_groups[1], 'lr': 10*args.res_lr, 'weight_decay': args.res_wt_dec},
        ], lr=args.res_lr, weight_decay=args.res_wt_dec, max_step=args.max_iters)
    elif ('vit' in args.network_type): 
        param_groups = model.trainable_parameters()
        optimizer = torch.optim.SGD([
            {'params': param_groups[0], 'lr': args.vit_lr, 'weight_decay': args.vit_wt_dec},
            {'params': param_groups[1], 'lr': 10*args.vit_lr, 'weight_decay': args.vit_wt_dec},    
        ], lr=args.vit_lr, weight_decay=args.vit_wt_dec, momentum=0.9)
    elif ('encdec' in args.network_type):
        param_groups = model.parameters()
        optimizer = torchutils.PolyOptimizer(
            param_groups, lr=args.encdec_lr, weight_decay=args.encdec_wt_dec, max_step=args.max_iters)
    else:
        raise("No Proper Optimizer!!!")
    return optimizer

def get_scheduler(args, optimizer):
    if ('vit' in args.network_type): 
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iters)
        return lr_scheduler
    else:
        return None