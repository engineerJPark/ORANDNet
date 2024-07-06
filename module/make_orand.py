import torch
from torch import multiprocessing, cuda
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import os
import time
from tqdm import tqdm

from util import imutils
from metadata.dataset import make_input_output
from PIL import Image

cudnn.enabled = True


def _work(args):
    # load cam npy
    img_ids = open(args.datalist).read().splitlines()

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        w, h = img.size[0], img.size[1]
        strided_size = imutils.get_strided_size((h, w), 4)
        
        or_dict, and_dict = make_input_output(img_id, args)
        or_cams = torch.from_numpy(or_dict['high_res'])
        and_cams = torch.from_numpy(and_dict['high_res'])
        valid_cat = or_dict['keys'] # coded class number && [0, 2, 11]

        if and_cams.size()[-2:] == (h, w) and or_cams.size()[-2:] == (h, w):
            pass
        else:
            print("have somewhat problem")
            # print(and_cams.shape)
            and_cams = np.asarray(
                F.interpolate(and_cams.unsqueeze(dim=0), size=(h, w), mode='bilinear').squeeze(dim=0)) # (2, 384, 384)
            or_cams = np.asarray(
                F.interpolate(or_cams.unsqueeze(dim=0), size=(h, w), mode='bilinear').squeeze(dim=0)) # (2, 384, 384)
        
        lowres_cam_and = F.interpolate(and_cams.unsqueeze(dim=0), size=strided_size, mode='bilinear', align_corners=False).squeeze(dim=0)
        lowres_cam_or = F.interpolate(or_cams.unsqueeze(dim=0), size=strided_size, mode='bilinear', align_corners=False).squeeze(dim=0)
        
        #################################################edit needed
        # save cams
        np.save(os.path.join(args.pred_dir, img_id + '_and.npy'),
                {"keys": valid_cat, "cam": np.asarray(lowres_cam_and), "high_res": np.asarray(and_cams)})
        np.save(os.path.join(args.pred_dir, img_id + '_or.npy'),
                {"keys": valid_cat, "cam": np.asarray(lowres_cam_or), "high_res": np.asarray(or_cams)})
        

def run_make_and(args):
    _work(args)
    torch.cuda.empty_cache()