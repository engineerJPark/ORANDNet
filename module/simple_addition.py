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


from lrp.baselines.ViT.ViT_LRP import deit_base_patch16_384 as vit_LRP
from lrp.baselines.ViT.ViT_explanation_generator import LRP
from metadata.dataset import load_img_id_list

cudnn.enabled = True

def _work(args):
    img_id_list = load_img_id_list(args.datalist)
    for _, id in enumerate(img_id_list):
        dict1 = np.load('./savefile/cam/result' + '/resnet/cam_npy/' + id + '_simpleaddition.npy', allow_pickle=True).item()
        dict2 = np.load('./savefile/cam/result' + '/vit/cam_npy/'  + id + '_simpleaddition.npy', allow_pickle=True).item()
        valid_cat = dict1['keys']
        
        cam1 = dict1['high_res']
        cam2 = dict2['high_res'] + 0.15
        
        size = (cam1.shape[1], cam1.shape[2])
        strided_size = imutils.get_strided_size(size, 4)
        
        ### CAM 합치기
        highres = (cam1 + cam2) / 2
        highres = torch.from_numpy(highres)
        highres = F.relu(highres) / (F.adaptive_max_pool2d(highres, (1, 1)) + 1e-5)
        # highres = (highres - highres.min()) / (highres.max() - highres.min()) ## min max
        highres = np.asarray(highres)
        strided_cam = np.asarray(F.interpolate(
                torch.from_numpy(highres).unsqueeze(dim=0), size=strided_size, mode='bilinear').squeeze(dim=0)) # strided size
        
        np.save(os.path.join(args.pred_dir, id + '_1.npy'), ### CAM for simple addition
                {"keys": valid_cat, "cam": strided_cam, "high_res": highres})
        

def run_simple_addition(args):
    print("SIMPLE ADDITION")
    # data_loader = get_dataloader(args) # load dataset
    
    _work(args)
    torch.cuda.empty_cache()