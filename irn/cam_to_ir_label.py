
import os
import numpy as np
import imageio

import torch
from torch import multiprocessing
from torch.utils.data import DataLoader

from metadata.dataset import VOC12ImageDataset
from util import torchutils, imutils
from tqdm import tqdm
from metadata.dataset import make_input_output


# def _work(infer_dataset, args):
#     infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=0, pin_memory=False)
    
def _work(process_id, infer_dataset, args):
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(databin, shuffle=False, num_workers=0, pin_memory=False)
    
    for iter, pack in tqdm(enumerate(infer_data_loader)):
        # for i in range(1, args.save_iter+1):
        img_name = pack['name'][0] # voc12.dataloader.decode_int_filename(pack['name'][0])
        img = pack['img'][0].numpy()
        
        # recent comment
        cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '_1.npy'), allow_pickle=True).item()
        # cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '_1.npy'), allow_pickle=True).item()
        
        cams = cam_dict['high_res'][1:] if 'crf' in args.network_type \
            else cam_dict['high_res']
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_fg_thres)
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.conf_bg_thres)
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '_1.png'), conf.astype(np.uint8))
        # imageio.imwrite(os.path.join(args.ir_label_out_dir, img_name + '_%d.png'%(i)), conf.astype(np.uint8))

def run(args):
    dataset = VOC12ImageDataset(args.datalist, voc12_root=args.data_root, img_normal=None, to_torch=False)
    dataset = torchutils.split_dataset(dataset, args.num_workers)
    
    # _work(dataset, args)
    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
    print(']')