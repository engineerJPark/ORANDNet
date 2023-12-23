import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio
from tqdm import tqdm

from metadata.dataset import VOC12ClassificationDatasetMSF
from util import torchutils, indexing
import os

cudnn.enabled = True


# def _work(model, dataset, args):
#     data_loader = DataLoader(dataset,
#                              shuffle=False, num_workers=args.num_workers, pin_memory=False)

def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin,
                             shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)
    
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    
    if args.dataset == 'coco':
        coco = True

    # with torch.no_grad():
    with torch.no_grad(), cuda.device(process_id):
        model.cuda()

        for iter, pack in tqdm(enumerate(data_loader)):
            img_name = pack['name'][0] # img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            height, width = pack['size'][0].item(), pack['size'][1].item()
            orig_img_size = np.array([height, width])

            edge, dp = model(pack['img'][0].cuda(non_blocking=True), coco=coco) # .cuda(non_blocking=True)

            cam_dict = np.load(args.cam_out_dir + '/' + img_name + '_1.npy', allow_pickle=True).item()

            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

            cam_downsized_values = torch.from_numpy(cams).cuda(non_blocking=True)
            
            rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

            rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :orig_img_size[0], :orig_img_size[1]]
            rw_up = rw_up / torch.max(rw_up)

            np.save(os.path.join(args.pred_dir, img_name + '_1.npy'), rw_up.cpu().numpy())

            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]
            
            if rw_pred.shape != (orig_img_size[0], orig_img_size[1]):
                print(orig_img_size)
                print(rw_pred.shape)
                raise ("stop")
            imageio.imsave(os.path.join(args.sem_seg_out_dir, 'pseudo_mask_' + img_name + '_1.png'), rw_pred.astype(np.uint8))


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False) # model.load_state_dict(torch.load(args.irn_weights_name, map_location=torch.device('cpu')), strict=False)
    model.eval()
    
    n_gpus = torch.cuda.device_count()
    dataset = VOC12ClassificationDatasetMSF(args.datalist, args.dataset,
                                            voc12_root=args.data_root,
                                            scales=(1.0,))
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")
    torch.cuda.empty_cache()

# def run(args):
#     model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
#     try:
#         model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
#     except:
#         model.load_state_dict(torch.load(args.irn_weights_name, map_location=torch.device('cpu')), strict=False)
#     model.eval()
#     dataset = VOC12ClassificationDatasetMSF(args.datalist,
#                                             args.dataset,
#                                             voc12_root=args.data_root,
#                                             scales=(1.0,))
#     _work(model, dataset, args)

#     torch.cuda.empty_cache()