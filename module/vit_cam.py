import numpy as np
import torch
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
from torch import multiprocessing, cuda

from module.model import get_model
from torch.utils.data import DataLoader
from module.dataloader import get_dataloader
from util import torchutils, imutils
from metadata.dataset import ClassificationDataset
from torchvision import transforms

from lrp.baselines.ViT.ViT_LRP import deit_base_patch16_384 as vit_LRP
from lrp.baselines.ViT.ViT_explanation_generator import LRP

# def vit_grad_cam(model, data_loader, args):

def vit_grad_cam(process_id, model, dataset, args):
    
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(databin, batch_size=1,
                            shuffle=False, num_workers=args.num_workers // n_gpus,
                            pin_memory=False, drop_last=False)

    attribution_generator = LRP(model)
    st = time.time()
    for iteration, pack in tqdm(enumerate(data_loader)):
        img_name = pack['name'][0]
        label = pack['label'][0] # one hot encoded
        valid_cat = torch.nonzero(label)[:, 0] # nonzero label index for all batch
        
        input_tensor = pack['img']  # Create an input tensor image for your model..
        size = pack['size']

        strided_size = imutils.get_strided_size(size, 4)
        
        grayscale_cam_low_res_list = []
        grayscale_cam_high_res_list = []
        
        # if 'simpleaddition' in args.network_type:
        #     for idx in range(len(valid_cat)):
        #         transformer_attribution = attribution_generator.generate_LRP(input_tensor.to(args.device),\
        #             method="transformer_attribution", index=valid_cat[idx]).detach()
        #         transformer_attribution = transformer_attribution.reshape(1, 32, 32)

        #         grayscale_cam = transformer_attribution.cpu().numpy()
                
        #         grayscale_cam_low_res = np.asarray(F.interpolate(
        #             torch.from_numpy(grayscale_cam).unsqueeze(dim=0), size=strided_size,
        #         mode='bilinear').squeeze(dim=0)) # strided size
        #         grayscale_cam_low_res_list.append(grayscale_cam_low_res)
                
        #         grayscale_cam_high_res = np.asarray(F.interpolate(
        #             torch.from_numpy(grayscale_cam).unsqueeze(dim=0), size=(size[0], size[1]),
        #         mode='bilinear').squeeze(dim=0)) # to original size
        #         grayscale_cam_high_res_list.append(grayscale_cam_high_res)
            
        #     grayscale_cam_low_res_stacked = np.concatenate(grayscale_cam_low_res_list, axis=0)
        #     grayscale_cam_high_res_stacked = np.concatenate(grayscale_cam_high_res_list, axis=0)
            
        #     np.save(os.path.join(args.pred_dir, img_name + '_simpleaddition.npy'), 
        #             {"keys": valid_cat, "cam": grayscale_cam_low_res_stacked, "high_res": grayscale_cam_high_res_stacked})
        
        # else:
        try: 
            for idx in range(len(valid_cat)):
                transformer_attribution = attribution_generator.generate_LRP(input_tensor.to(args.device),\
                    method="transformer_attribution", index=valid_cat[idx]).detach()
                transformer_attribution = transformer_attribution.reshape(1, 32, 32)
                transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
                # # compensate the diff of resnet ####################################
                # # transfer to grayscale cam ####################################
                grayscale_cam = transformer_attribution.cpu().numpy()
                
                grayscale_cam_low_res = np.asarray(F.interpolate(
                    torch.from_numpy(grayscale_cam).unsqueeze(dim=0), size=strided_size,
                mode='bilinear').squeeze(dim=0)) # strided size
                grayscale_cam_low_res_list.append(grayscale_cam_low_res)
                
                grayscale_cam_high_res = np.asarray(F.interpolate(
                    torch.from_numpy(grayscale_cam).unsqueeze(dim=0), size=(size[0], size[1]),
                mode='bilinear').squeeze(dim=0)) # to original size
                grayscale_cam_high_res_list.append(grayscale_cam_high_res)
            
            # print(grayscale_cam_low_res_list[0].shape)
            # print(len(grayscale_cam_low_res_list))
            
            grayscale_cam_high_res_stacked = np.concatenate(grayscale_cam_high_res_list, axis=0)
            grayscale_cam_low_res_stacked = np.concatenate(grayscale_cam_low_res_list, axis=0)
            
            np.save(os.path.join(args.pred_dir, img_name + '_1.npy'), 
                    {"keys": valid_cat, "cam": grayscale_cam_low_res_stacked, "high_res": grayscale_cam_high_res_stacked})
        except: 
            print(input_tensor.shape)
            print(img_name)
            print(valid_cat)
            print(len(grayscale_cam_high_res_list))
            print(valid_cat)
            print(input_tensor.shape)
            # print(len(grayscale_cam_low_res_list))
            # print(len(grayscale_cam_high_res))


def run_vit_cam(args):
    print("ViT CAM")
    model = get_model(args) # model = vit_LRP(pretrained=True).cuda()
    model.to(args.device).eval()
    
    n_gpus = torch.cuda.device_count()
    dataset = ClassificationDataset(
        args.dataset,
        args.datalist,
        img_root=args.data_root,
        transform=transforms.Compose([
        np.asarray,
        imutils.Normalize(),
        imutils.HWC_to_CHW,
        torch.from_numpy,
        transforms.Resize((args.crop_size,args.crop_size))
    ]))
    dataset = torchutils.split_dataset(dataset, n_gpus)
    
    print("[", end='')
    multiprocessing.spawn(vit_grad_cam, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")
    torch.cuda.empty_cache()
    
    # data_loader = get_dataloader(args) # load dataset
    # vit_grad_cam(model, data_loader, args)