import torch
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm
from util import imutils

from util.imutils import _crf_with_alpha
from metadata.dataset import load_img_id_list, load_img_label_list_from_npy

    
def _work(args):

    img_id_list = load_img_id_list(args.datalist)
    label_list = load_img_label_list_from_npy(img_id_list, args.dataset)
    
    os.makedirs(args.crf_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir + 'default/crf', exist_ok=True)
    
    for idx in tqdm(range(len(img_id_list))):
        img_id = img_id_list[idx]
        label = np.array(label_list[idx])
        
        valid_cat = np.nonzero(label)[0]
        keys = np.pad(valid_cat + 1, (1, 0), mode='constant')
        img = np.asarray(Image.open(os.path.join(args.data_root, img_id + '.jpg')).convert("RGB"))
        
        ## normal CAM refine
        cam = np.load(os.path.join(args.cam4crf_dir, img_id + '_1.npy'), allow_pickle=True).item()
        cam_crf, cam_crf_np = _crf_with_alpha(img, cam, alpha=args.alpha, t=args.t) # get rid of background 
                
        np.save(os.path.join(args.crf_out_dir, img_id + '_1.npy'), cam_crf)
        
        cam_crf_int = np.argmax(cam_crf_np, axis=0)
        cam_crf_int = np.array(keys[cam_crf_int], dtype=np.uint8)
        cam_crf_int = Image.fromarray(cam_crf_int)
        if args.network_type == 'crf': ############
            cam_crf_int.save(args.crf_sem_seg_out_dir + '/pseudo_mask_%s_1.png' % (img_id))
        
        


def run_crf_refine(args):
    print("CRF INFERENCE")   
    _work(args)
    torch.cuda.empty_cache()