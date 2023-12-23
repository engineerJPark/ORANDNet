import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.backends import cudnn
from PIL import Image

import numpy as np
import importlib
import os
from tqdm import tqdm
from module.model import get_model
from module.dataloader import get_dataloader
from util import torchutils, imutils
from metadata.dataset import Or2AndCamDataset # ClassificationDataset, ClassificationDataset_MultiScale

torch.backends.cudnn.enabled = True


def cam2xor(cam1, cam2):
    '''
    cam1, cam2 should be numpy tensor
    3 dimension
    '''
    outcam = cam1 + cam2 - cam1 * cam2 * 2
    return outcam


@torch.no_grad()
def _work(process_id, model, dataset, args):
    print("SINGLE INFERENCE")
    
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin, batch_size=1,
                            shuffle=False, num_workers=args.num_workers // n_gpus,
                            pin_memory=args.pin_memory, drop_last=False)

    # with torch.no_grad():
    for iteration, pack in tqdm(enumerate(data_loader)): # 1 batch, c gt labels
        input_cam_highres = pack['input_cam_highres'].to(args.device, non_blocking=True) # (b,c,h,w) b=1
        img = pack['img'].to(args.device, non_blocking=True)
        img_name = pack['name'][0]
        label = pack['label'][0] # one hot encoded
        valid_cat = torch.nonzero(label)[:, 0] # start with 0, code page labels
        size = pack['size']
        
        strided_size = imutils.get_strided_size(size, 4)
        
        ##############################Iteration start
        # high res cam
        outputs = list()
        for idx in range(input_cam_highres.shape[1]): # for each gt channel
            outputs.append(model(input_cam_highres[:, idx].unsqueeze(dim=0)).squeeze(dim=0)) # output is (1, H, W)
        outputs = [F.relu(o) for o in outputs] # (C, H, W)
        outputs = torch.cat(outputs, dim=0).unsqueeze(dim=0) # (1, C, H, W)
        outputs /= F.adaptive_max_pool2d(outputs, (1, 1)) + 1e-5 # Normalization
        highres_cam = outputs.squeeze(dim=0)
        
        # low res cam
        lowres_cam = F.interpolate(highres_cam.unsqueeze(dim=0), size=strided_size, mode='bilinear', align_corners=False).squeeze(dim=0)

        np.save(os.path.join(args.pred_dir, img_name + '_1.npy'), \
            {"keys": valid_cat, "cam": lowres_cam.cpu().numpy(), "high_res": highres_cam.cpu().numpy()})
        
        input_cam_highres = highres_cam.unsqueeze(dim=0)



def run_encdec_pred(args):
    model = get_model(args)
    model.eval()
    model.to(args.device)

    dataset = Or2AndCamDataset(args.dataset, args.datalist, img_root=args.data_root, args=args, network_type=args.network_type)
    n_gpus = torch.cuda.device_count()
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")
    torch.cuda.empty_cache()