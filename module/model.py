import importlib
import os
import torch
from torchvision.models import vgg16

def get_model(args):
    if args.network_type == 'resnet50' or args.network_type == 'resnet50_coco':
        method = getattr(importlib.import_module(args.network + '.resnet50_cam'), 'Net')
        model = method(args)
    elif args.network_type == 'resnet50_cam' or args.network_type == 'resnet50_cam_simpleaddition':
        method = getattr(importlib.import_module(args.network+ '.resnet50_cam'), 'CAM')
        model = method(args)
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'resnet50_cam.pth')), strict=True)
    elif args.network_type == 'resnet50_cam_coco':
        method = getattr(importlib.import_module(args.network+ '.resnet50_cam'), 'CAM')
        model = method(args)
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'resnet50_cam_coco.pth')), strict=True)
    elif args.network_type == 'vit' or args.network_type == 'vit_coco':
        method = getattr(importlib.import_module(args.network + '.vit'), 'vit')
        model = method(args)
    elif args.network_type == 'vit_cam' or args.network_type == 'vit_cam_simpleaddition':
        method = getattr(importlib.import_module(args.network+ '.vit'), 'vit')
        model = method(args)
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'vit_cam.pth')), strict=True)
    elif args.network_type == 'vit_cam_coco':
        method = getattr(importlib.import_module(args.network+ '.vit'), 'vit')
        model = method(args)
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'vit_cam_coco.pth')), strict=True)
    elif args.network_type == 'encdec' or args.network_type == 'encdec_coco':
        method = getattr(importlib.import_module(args.network+ '.encdec'), 'EncDec')
        model = method()
        model.copy_params_from_vgg16(vgg16(pretrained=True))
    elif args.network_type == 'encdec_pred':
        method = getattr(importlib.import_module(args.network+ '.encdec'), 'EncDec')
        model = method()
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'encdec.pth')), strict=True)
    elif args.network_type == 'encdec_pred_coco':
        method = getattr(importlib.import_module(args.network+ '.encdec'), 'EncDec')
        model = method()
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'encdec_coco.pth')), strict=True)
    elif args.network_type == 'encdec_amnmct' or args.network_type == 'encdec_amnmct_coco':
        method = getattr(importlib.import_module(args.network+ '.encdec'), 'EncDec')
        model = method()
        model.copy_params_from_vgg16(vgg16(pretrained=True))
    elif args.network_type == 'encdec_pred_amnmct':
        method = getattr(importlib.import_module(args.network+ '.encdec'), 'EncDec')
        model = method()
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'encdec_amnmct.pth')), strict=True)
    elif args.network_type == 'encdec_pred_amnmct_coco':
        method = getattr(importlib.import_module(args.network+ '.encdec'), 'EncDec')
        model = method()
        model.load_state_dict(torch.load(os.path.join(args.weight_root, 'encdec_amnmct_coco.pth')), strict=True)
    elif args.network_type == 'irn_simpleaddition':
        method = getattr(importlib.import_module(args.network + '.resnet50_cam'), 'Net')
        model1 = method(args)
        model1.load_state_dict(torch.load(os.path.join(args.weight_root, 'resnet50_cam.pth')), strict=True)
        
        method = getattr(importlib.import_module(args.network + '.vit'), 'vit')
        model2 = method(args)
        model2.load_state_dict(torch.load(os.path.join(args.weight_root, 'vit_cam.pth')), strict=True)
        return model1, model2
    else:
        raise("No model like this!!!")
    return model