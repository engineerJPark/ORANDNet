import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
from module.dataloader import get_dataloader
import time
from tqdm import tqdm
from metadata.dataset import make_input_output
from util import imutils


CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

VOC_COLORMAP = np.array(
                [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], 
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
                [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
                )

cudnn.enabled = True

def get_labels(label_file):
    idx2num = list()
    idx2label = list()
    for line in open(label_file).readlines():
        num, label = line.strip().split()
        idx2num.append(num)
        idx2label.append(label)
    return idx2num, idx2label




def _work(args):
    # load cam npy
    img_ids = open(args.datalist).read().splitlines()

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        w, h = img.size[0], img.size[1]

        if args.network_type == 'amn':
            pred_path = os.path.join(args.pred_dir, img_id + '.npy')
            cam_dict = np.load(pred_path, allow_pickle=True).item()
            cams = torch.from_numpy(cam_dict['high_res'])[1:]
            valid_cat = cam_dict['keys'] # coded class number && [0, 2, 11]
        elif args.network_type == 'mctformer':
            pred_path = os.path.join(args.pred_dir, img_id + '.npy')
            cam_dict = np.load(pred_path, allow_pickle=True).item()
            valid_cat = list(cam_dict.keys())
            cams = torch.tensor(list(cam_dict.values()))

        else:
            pred_path = os.path.join(args.pred_dir, img_id + '_1.npy')
            cam_dict = np.load(pred_path, allow_pickle=True).item() # (2, 281, 500) (2, 384, 384)
            cams = torch.from_numpy(cam_dict['high_res'])
            valid_cat = cam_dict['keys'] # coded class number && [0, 2, 11]
            
        if cams.size()[-2:] == (h, w):
            pass
        else:
            print("have somewhat problem")
            cams = np.asarray(
                F.interpolate(cams.unsqueeze(dim=0), size=(h, w), mode='bilinear').squeeze(dim=0)) # (2, 384, 384)
        cam_img_pil = []
        for channel_idx in range(cams.shape[0]): # cam img for each class + coloring
            cam_img_pil.append(Image.fromarray(np.uint8(cm.jet(cams[channel_idx, ...]) * 255)).convert("RGB"))
        for channel_idx in range(cams.shape[0]): # superpose on image
            alphaComposited = Image.blend(img, cam_img_pil[channel_idx], 0.70)
            alphaComposited.save(args.draw_dir +'/cam_%s_%s_.png' % (img_id, CAT_LIST[valid_cat[channel_idx]]))

def _work_orand(args):
    # load cam npy
    img_ids = open(args.datalist).read().splitlines()

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        w, h = img.size[0], img.size[1]
        
        or_dict, and_dict = make_input_output(img_id, args)
        or_cams = torch.from_numpy(or_dict['high_res'])
        and_cams = torch.from_numpy(and_dict['high_res'])
        valid_cat = or_dict['keys'] # coded class number && [0, 2, 11]
        
        size = (w, h)
        strided_size = imutils.get_strided_size(size, 4)
        or_lowres_cam = F.interpolate(or_cams.unsqueeze(dim=0), size=strided_size, mode='bilinear', align_corners=False).squeeze(dim=0)
        and_lowres_cam = F.interpolate(and_cams.unsqueeze(dim=0), size=strided_size, mode='bilinear', align_corners=False).squeeze(dim=0)
        np.save(os.path.join(args.pred_dir, 'or_' + img_id + '.npy'), \
            {"keys": valid_cat, 
             "cam": or_lowres_cam.cpu().numpy(), 
             "high_res": or_cams.cpu().numpy()})
        np.save(os.path.join(args.pred_dir, 'and_' + img_id + '.npy'), \
            {"keys": valid_cat, 
             "cam": and_lowres_cam.cpu().numpy(), 
             "high_res": and_cams.cpu().numpy()})

        if or_cams.size()[-2:] == (h, w):
            pass
        else:
            print("have somewhat problem")
            # print(or_cams.shape)
            or_cams = np.asarray(
                F.interpolate(or_cams.unsqueeze(dim=0), size=(h, w), mode='bilinear').squeeze(dim=0)) # (2, 384, 384)
        cam_img_pil = []
        for channel_idx in range(or_cams.shape[0]): # cam img for each class + coloring
            cam_img_pil.append(Image.fromarray(np.uint8(cm.jet(or_cams[channel_idx, ...]) * 255)).convert("RGB"))
        for channel_idx in range(or_cams.shape[0]): # superpose on image
            alphaComposited = Image.blend(img, cam_img_pil[channel_idx], 0.70)
            alphaComposited.save(args.draw_dir +'/or_cam_%s_%s_.png' % (img_id, CAT_LIST[valid_cat[channel_idx]]))
            
        if and_cams.size()[-2:] == (h, w):
            pass
        else:
            print("have somewhat problem")
            # print(and_cams.shape)
            and_cams = np.asarray(
                F.interpolate(and_cams.unsqueeze(dim=0), size=(h, w), mode='bilinear').squeeze(dim=0)) # (2, 384, 384)
        cam_img_pil = []
        for channel_idx in range(and_cams.shape[0]): # cam img for each class + coloring
            cam_img_pil.append(Image.fromarray(np.uint8(cm.jet(and_cams[channel_idx, ...]) * 255)).convert("RGB"))
        for channel_idx in range(and_cams.shape[0]): # superpose on image
            alphaComposited = Image.blend(img, cam_img_pil[channel_idx], 0.70)
            alphaComposited.save(args.draw_dir +'/and_cam_%s_%s_.png' % (img_id, CAT_LIST[valid_cat[channel_idx]]))
            
def _work_label(args):
    img_ids = open(args.datalist).read().splitlines()

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        w, h = img.size[0], img.size[1]

        # pred_path = os.path.join(args.pred_dir, 'pseudo_mask_' + img_id + '_1.png')
        pred_path = os.path.join(args.pred_dir, img_id + '.png')
        cams = Image.open(pred_path)
        cams = np.asarray(cams)
        
        cam_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[cams])).convert("RGB") # cam_img_pil = VOC_COLORMAP[cams]
        alphaComposited = Image.blend(img, cam_img_pil, 0.70)
        alphaComposited.save(args.draw_dir +'/cam_%s_1.png' % (img_id))

def run_draw(args):
    n_gpus = torch.cuda.device_count()

    if 'irn' in args.network_type:
        _work_label(args)
    elif 'orand' in args.network_type:
        _work_orand(args)
    else:
        _work(args)
    torch.cuda.empty_cache()