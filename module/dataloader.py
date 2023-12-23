import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from metadata.dataset import ClassificationDataset, ClassificationDataset_MultiScale, Or2AndCamDataset, pil2np_3dim
from util import imutils


def get_dataloader(args):
    if 'resnet50_cam' in args.network_type:
        print("Multiscale CAM Dataset")
        train_dataset = ClassificationDataset_MultiScale(
            args.dataset,
            args.datalist,
            img_root=args.data_root
            )
        train_loader = DataLoader(train_dataset, batch_size=1,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, drop_last=False)
    elif 'vit_cam' in args.network_type:
        print("ViT CAM Dataset")       
        train_dataset = ClassificationDataset(
            args.dataset,
            args.datalist,
            img_root=args.data_root,
            transform=transforms.Compose([
            np.asarray,
            imutils.Normalize(),
            transforms.Lambda(pil2np_3dim),
            imutils.HWC_to_CHW,
            torch.from_numpy,
            transforms.Resize((args.crop_size,args.crop_size))
        ]))
        train_loader = DataLoader(train_dataset, batch_size=1,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, drop_last=False)
    elif 'encdec_pred' in args.network_type:
        print("Or2AndCamDataset Dataset")
        train_dataset = Or2AndCamDataset(args.dataset, args.datalist, img_root=args.data_root, args=args, network_type=args.network_type)
        train_loader = DataLoader(train_dataset, batch_size=1,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, drop_last=False)
            
    elif  'resnet50' in args.network_type or 'vit' in args.network_type or args.network_type == 'orand':
        print("Plain Classification Dataset")
        train_dataset = ClassificationDataset(
            args.dataset,
            args.datalist,
            img_root=args.data_root,
            transform=transforms.Compose([
            imutils.RandomResizeLong(args.resize_size[0], args.resize_size[1]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.Lambda(pil2np_3dim),
            imutils.Normalize(),
            imutils.RandomCrop(args.crop_size),
            imutils.HWC_to_CHW,
            torch.from_numpy
        ]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, drop_last=True)
    
    elif 'encdec_pred' in args.network_type:
        print("Or2AndCamDataset Prediction")
        train_dataset = Or2AndCamDataset(args.dataset, args.datalist, img_root=args.data_root, args=args, network_type=args.network_type)
        train_loader = DataLoader(train_dataset, batch_size=1,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, drop_last=False)
    elif 'encdec' in args.network_type:
        print("Or2AndCamDataset Train")
        train_dataset = Or2AndCamDataset(args.dataset, args.datalist, img_root=args.data_root, args=args, network_type=args.network_type, training=True)
        train_loader = DataLoader(train_dataset, batch_size=1,
                                shuffle=True, num_workers=args.num_workers,
                                pin_memory=args.pin_memory, drop_last=True)
    else:
        raise Exception("No appropriate train type")
    return train_loader