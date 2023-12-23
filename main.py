import os
import torch
import argparse

from util import pyutils
import numpy as np
import random

random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


_NUM_CLASSES = {'voc12': 20, 'coco': 80}

def get_arguments():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", default='./dataset/VOCdevkit/VOC2012/JPEGImages', required=True, type=str) # 'dataset/VOCdevkit/VOC2012/JPEGImages'
    parser.add_argument("--dataset", default='voc12', type=str)
    parser.add_argument("--datalist", default="metadata/voc12/train.txt", type=str)
    parser.add_argument("--vallist", default="metadata/voc12/val.txt", type=str)
    parser.add_argument('--gt_dir', required=True, type=str)
    parser.add_argument('--pred_dir', required=True, type=str) # save npy cam
    parser.add_argument('--draw_dir', required=True, type=str)
    parser.add_argument('--pred_dir1', type=str) # save npy cam
    parser.add_argument('--pred_dir2', type=str) # save npy cam
    parser.add_argument('--draw_dir1', type=str)
    parser.add_argument('--draw_dir2', type=str)
    # parser.add_argument('--save_path', required=True, type=str) # save log
    parser.add_argument("--cam_out_root", default='./savefile/cam/result', type=str)
    # parser.add_argument("--sem_seg_out_dir", default='./dataset/PseudoMask/default', type=str)
    parser.add_argument("--log_root", default='./savefile/log')
    parser.add_argument("--crop_size", default=512, type=int) ## 448 384 224
    parser.add_argument("--resize_size", default=(256, 512), type=int, nargs='*')
    parser.add_argument("--pin_memory", default=False, type=bool)
    parser.add_argument("--disable_gpu", default=False, type=bool)
    
    # network selection : resnet50 or vit
    parser.add_argument("--network_type", type=str) # resnet_cam, vit, vit_cam
    parser.add_argument("--network", default='network', type=str, help='network folder name') # need to be change by real name...
    
    # train pass
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--max_iters", default=None, type=int)
    parser.add_argument("--num_workers", default=8, type=int) # 1, os.cpu_count()//4
    parser.add_argument("--weight_root", default='./savefile/weight')
    
    ### resnet
    parser.add_argument("--res_lr", default=0.01, type=float) # optimizer
    parser.add_argument("--res_wt_dec", default=5e-4, type=float)
    
    ### vit
    # Optimizer Learning rate schedule parameters
    parser.add_argument("--vit_lr", default=0.01, type=float) # 0.001
    parser.add_argument("--vit_wt_dec", default=1e-5, type=float) # 0 1e-5
    parser.add_argument("--clip", default=1, type=float, help='gradient clipping norm')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    
    # encdec training arguments
    parser.add_argument("--encdec_lr", default=0.001, type=float) # 0.01
    parser.add_argument("--encdec_wt_dec", default=1e-4, type=float) # 4e-5
    
    
    # Inference : CAM making pass
    parser.add_argument("--thres", default=0.25, type=float)
    
    # Segmentation stage
    # parser.add_argument("--seg_thres", default=0.20, type=float)
    parser.add_argument("--deeplab_lr", default=2.5e-4, type=float) # 0.003  # 0.01
    parser.add_argument("--deeplab_wt_dec", default=5.0e-4, type=float) # 1e-4
    
    ## CRF
    parser.add_argument("--cam4crf_dir", default="./savefile/cam/result/encdec/cam_npy", type=str)
    parser.add_argument("--crf_out_dir", default="./savefile/cam/result/crf/encdec/cam_npy", type=str)
    parser.add_argument("--alpha", default=4, type=float) # 32
    parser.add_argument("--t", default=10, type=float)
    parser.add_argument("--crf_sem_seg_out_dir", default="./dataset/PseudoMask/default/crf", type=str)
    
    ## IRN parameters
    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.30, type=float) # 0.45 0.30
    parser.add_argument("--conf_bg_thres", default=0.05, type=float) # 0.15 0.05

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="irn.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=16, type=int) # 32
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)
    parser.add_argument("--irn_datalist", default="metadata/voc12/train_aug_for_irn.txt", type=str)
    
    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.30, type=float) # 0.25
    parser.add_argument("--sem_seg_bg_thres", default=0.30, type=float) # 0.25
    parser.add_argument("--fg_seg_bg_thres", default=0.30, type=float) # 0.25
    parser.add_argument("--bg_seg_bg_thres", default=0.05, type=float) # 0.25
    
    # Output Path
    parser.add_argument("--irn_weights_name", default="./savefile/weight/resnet50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="./savefile/cam/result/encdec/cam_npy", type=str)
    parser.add_argument("--ir_label_out_dir", default="./dataset/ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="./dataset/PseudoMask/default/irn", type=str)
    
    # pass activation
    parser.add_argument("--train_pass", default=False, type=bool)
    parser.add_argument("--make_cam_pass", default=False, type=bool)
    parser.add_argument("--encdec_train_pass", default=False, type=bool)
    parser.add_argument("--encdec_pred_pass", default=False, type=bool)
    parser.add_argument("--simple_addition_pass", default=False, type=bool)
    parser.add_argument("--make_orand_label_pass", default=False, type=bool)
    parser.add_argument("--eval_pass", default=False, type=bool)
    parser.add_argument("--cam_to_ir_label", default=False, type=bool)
    parser.add_argument("--cam_to_ir_label_simpleaddition", default=False, type=bool)
    parser.add_argument("--irn_pass", default=False, type=bool)
    parser.add_argument("--make_sem_seg_pass", default=False, type=bool)
    parser.add_argument("--make_orand_sem_seg_pass", default=False, type=bool)
    parser.add_argument("--irn_eval_pass", default=False, type=bool)
    parser.add_argument("--crf_pass", default=False, type=bool)
    parser.add_argument("--crf_eval_pass", default=False, type=bool)
    parser.add_argument("--draw_pass", default=False, type=bool)
    parser.add_argument("--and_pass", default=False, type=bool)
    
    ################################################

    args = parser.parse_args()
    args.num_classes = _NUM_CLASSES[args.dataset]
    args.scales = (0.5, 1.0, 1.5, 2.0)
    
    return args

def main():
    # get arguments
    args = get_arguments()
    args.scales = (0.5, 1.0, 1.5, 2.0)

    # args.device = 'cpu'
    # args.use_cuda = False
    if not args.disable_gpu and torch.cuda.is_available(): 
        print('CUDA MODE')
        args.n_gpus = torch.cuda.device_count()
        args.device = 'cuda'
        args.use_cuda = True
        args.device_list = list()
        for i in range(args.n_gpus):
            args.device_list.append('cuda:%d'%(i))
    else:
        print('CPU MODE')
        args.device = 'cpu'
        args.use_cuda = False
    
    # set log & path
    os.makedirs(args.log_root, exist_ok=True)
    os.makedirs(args.gt_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    os.makedirs(args.draw_dir, exist_ok=True)
    os.makedirs(args.weight_root, exist_ok=True)
    os.makedirs('./savefile/weight/', exist_ok=True)
    os.makedirs('./savefile/coco/weight/', exist_ok=True)
    
    
    pyutils.Logger(os.path.join(args.log_root, 'run_%s.log'% (args.network_type)))
    print(vars(args))
    
    if args.train_pass is True:
        from module.train import run_train
        timer = pyutils.Timer('module.train:')
        run_train(args)
        
    if args.make_cam_pass is True:
        if 'resnet' in args.network_type or 'vgg' in args.network_type or 'res' in args.network_type:
            from module.make_cam import run_make_cam
            timer = pyutils.Timer('module.make_cam:')
            run_make_cam(args)
        elif 'vit' in args.network_type:
            from module.vit_cam import run_vit_cam
            timer = pyutils.Timer('module.vit_cam:')
            run_vit_cam(args)
    
    if args.encdec_train_pass is True:
        from module.encdec_train import run_encdec
        timer = pyutils.Timer('module.run_encdec:')
        run_encdec(args)
        
    if args.encdec_pred_pass is True:
        from module.encdec_pred import run_encdec_pred
        timer = pyutils.Timer('module.encdec_pred:')
        run_encdec_pred(args)

    if args.simple_addition_pass is True:
        from module.simple_addition import run_simple_addition
        timer = pyutils.Timer('module.simple_addition:')
        run_simple_addition(args)
        

###########################################################

    if args.crf_pass is True:
        from crf.crf_refine import run_crf_refine
        timer = pyutils.Timer('module.crf_refine:')
        run_crf_refine(args)
    
###########################################################

    if args.cam_to_ir_label is True:
        from irn import cam_to_ir_label
        timer = pyutils.Timer('module.cam_to_ir_label:')
        cam_to_ir_label.run(args)
        
    if args.cam_to_ir_label_simpleaddition is True:
        from irn import cam_to_ir_label_simple_addition
        timer = pyutils.Timer('module.cam_to_ir_label:')
        cam_to_ir_label_simple_addition.run(args)
    
    if args.irn_pass is True:
        from irn.train_irn import run_train_irn
        timer = pyutils.Timer('module.train_irn:')
        run_train_irn(args)

    if args.make_sem_seg_pass is True:
        from irn import make_sem_seg_labels
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        make_sem_seg_labels.run(args)

###########################################################

    if args.eval_pass is True:
        from module.eval import eval
        timer = pyutils.Timer('module.eval:')
        eval(args)
        
    if args.crf_eval_pass is True:
        from crf.eval_crf import eval_crf
        timer = pyutils.Timer('module.eval_crf:')
        eval_crf(args)
        
    if args.irn_eval_pass is True:
        from irn.eval_irn import eval_irn
        timer = pyutils.Timer('module.eval_irn:')
        eval_irn(args)
        
###########################################################

    if args.draw_pass is True:
        from module.draw import run_draw
        timer = pyutils.Timer('module.draw:')
        run_draw(args)
        
    if args.and_pass is True:
        from module.make_orand import run_make_and
        timer = pyutils.Timer('module.run_make_and:')
        run_make_and(args)
    
if __name__ == '__main__':
    main()