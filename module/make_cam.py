import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import time
from tqdm import tqdm

from module.model import get_model
from module.dataloader import get_dataloader
from util import torchutils, imutils
from metadata.dataset import ClassificationDataset_MultiScale

cudnn.enabled = True


# def _work(process_id, model, dataset, args):
def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin, batch_size=1,
                            shuffle=False, num_workers=args.num_workers // n_gpus,
                            pin_memory=False, drop_last=False)
    
    with torch.no_grad():
        
        st = time.time()
        for iteration, pack in tqdm(enumerate(data_loader)):
            img_name = pack['name'][0]
            label = pack['label'][0] # one hot encoded
            valid_cat = torch.nonzero(label)[:, 0] # start with 0, code page labels
            size = pack['size']
            
            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)
            
            if 'simpleaddition' in args.network_type:
                # input for the model should be given in 2 batches
                outputs = [model.forward_nothres(img[0].to(args.device)) for img in pack['img']] # img is given by list, img is [1, 2, 3, 384, 384] -> img[0] = [2, 3, 384, 384]]
                
                # small size cam
                strided_cam = torch.sum(torch.stack(
                    [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] 
                    for o in outputs]), 0)
                strided_cam_selected = strided_cam[valid_cat]
                # strided_cam_selected /= F.adaptive_max_pool2d(strided_cam_selected, (1, 1)) + 1e-5
                
                # original image size cam
                highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs] # [c,h,w] -> [1,c,h,w]
                highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]] # [c,h,w]
                highres_cam_selected = highres_cam[valid_cat] # [c,h,w] c : gt class
                # highres_cam_selected /= F.adaptive_max_pool2d(highres_cam_selected, (1, 1)) + 1e-5 # Normalization [c,h,w]
                
                # save cams
                np.save(os.path.join(args.pred_dir, img_name + '_simpleaddition.npy'), 
                        {"keys": valid_cat, "cam": strided_cam_selected.cpu().numpy(), "high_res": highres_cam_selected.cpu().numpy()})
                
            else: # normal situation
                # input for the model should be given in 2 batches
                outputs = [model(img[0].to(args.device)) for img in pack['img']] # img is given by list, img is [1, 2, 3, 384, 384] -> img[0] = [2, 3, 384, 384]]
                
                # small size cam
                strided_cam = torch.sum(torch.stack(
                    [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] 
                    for o in outputs]), 0)
                strided_cam_selected = strided_cam[valid_cat]
                strided_cam_selected /= F.adaptive_max_pool2d(strided_cam_selected, (1, 1)) + 1e-5
                
                # original image size cam
                highres_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                            mode='bilinear', align_corners=False) for o in outputs] # [c,h,w] -> [1,c,h,w]
                highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, :size[0], :size[1]] # [c,h,w]
                highres_cam_selected = highres_cam[valid_cat] # [c,h,w] c : gt class
                highres_cam_selected /= F.adaptive_max_pool2d(highres_cam_selected, (1, 1)) + 1e-5 # Normalization [c,h,w]
                
                # save cams
                np.save(os.path.join(args.pred_dir, img_name + '_1.npy'), 
                        {"keys": valid_cat, "cam": strided_cam_selected.cpu().numpy(), "high_res": highres_cam_selected.cpu().numpy()})


def run_make_cam(args):
    print("RESNET CAM")
    model = get_model(args)
    model.eval()
    model.to(args.device)
    n_gpus = torch.cuda.device_count()
    dataset = ClassificationDataset_MultiScale(
        args.dataset,
        args.datalist,
        img_root=args.data_root
        )
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')
    torch.cuda.empty_cache()