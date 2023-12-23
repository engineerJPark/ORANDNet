import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from util import pyutils

VOC_COLORMAP = np.array(
                [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], 
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], 
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], 
                [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]
                )

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # mask = (label_true >= 0) & (label_true < self.num_classes)
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        # recall = np.nanmean(recall)
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        # precision = np.nanmean(precision)
        TP = np.diag(self.hist)
        TN = self.hist.sum(axis=1) - np.diag(self.hist)
        FP = self.hist.sum(axis=0) - np.diag(self.hist)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_classes), iu))

        return acc, recall, precision, TP, TN, FP, cls_iu, mean_iu, fwavacc

def get_labels(label_file):
    idx2num = list()
    idx2label = list()
    for line in open(label_file).readlines():
        num, label = line.strip().split()
        idx2num.append(num)
        idx2label.append(label)
    return idx2num, idx2label


def evalNdraw(args, rank=100, reverse=False):
    '''
    True :big val first
    '''
    
    # best_thres = finding_best_thres(args)
    # img_ids = open(args.datalist[:-4] + '_aug.txt').read().splitlines()
    img_ids = open(args.datalist[:-4] + '.txt').read().splitlines()
    
    # get arguments
    idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))
    st = time.time()

    miou_img_id = dict()
    for idx, img_id in tqdm(enumerate(img_ids)):
        mIOU = IOUMetric(num_classes=args.num_classes)
        
        gt_path = os.path.join(args.gt_dir, img_id + '.png')
        gt = Image.open(gt_path) # HW
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        
        # pred_path = os.path.join(args.pred_dir, img_id + '.png')
        pred_path = os.path.join(args.pred_dir, img_id + '.png')
        pred = Image.open(pred_path)
        pred = np.asarray(pred)
        
        mIOU.add_batch(pred, gt) # input to mIoU class
        acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()
        miou_img_id[img_id] = miou
        
        del mIOU
        
    reverse_miou_img_id = {v:k for k,v in miou_img_id.items()}
    
    miou_list = []
    img_id_list = []
    for key, value in sorted(miou_img_id.items(), key = lambda item: item[1], reverse=reverse): # low first ,,, reverse = True
        img_id_list.append(key)
        miou_list.append(value)
    miou_list = miou_list[:rank]
    img_id_list = img_id_list[:rank]
    
    print(miou_list)
    print(img_id_list)
    
    print("mIoU of Top 5")
    print(miou_list[0])
    print(miou_list[1])
    print(miou_list[2])
    print(miou_list[3])
    print(miou_list[4])

    _work_label(args, img_id_list)

def _work_label(args, img_id_list):
    img_ids = img_id_list

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        w, h = img.size[0], img.size[1]

        pred_path = os.path.join(args.pred_dir, img_id + '.png')
        cams = Image.open(pred_path)
        cams = np.asarray(cams)
        
        cam_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[cams])).convert("RGB") # cam_img_pil = VOC_COLORMAP[cams]
        alphaComposited = Image.blend(img, cam_img_pil, 0.70)
        alphaComposited.save(args.draw_dir +'/rank%d_%s.png' % (idx + 1, img_id))
        
        gt_path = os.path.join(args.gt_dir, img_id + '.png')
        gt = Image.open(gt_path) # HW
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        gt[(gt==255)] = 0
        
        # print(gt[(gt==255)])
        # print(gt==255)
        # print(np.unique(gt))
        
        gt_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[gt])).convert("RGB") # gt_img_pil = VOC_COLORMAP[cams]
        alphaComposited = Image.blend(img, gt_img_pil, 0.70)
        alphaComposited.save(args.draw_dir +'/rank%d_%s_GT.png' % (idx + 1, img_id))
        
    
    print("Done!!!!!!!!!!! Lets Compare!!!!!!!!!!!!!!!!!!!")
    
########################################################################
########################################################################
########################Compare 2 model Result##########################
########################################################################
########################################################################

def evalNdrawNcomparison(args, rank=100, reverse=False):
    '''
    True :big val first
    '''
    
    prefix = 'pseudo_mask_'
    subfix = '_1.png'
    
    # img_ids = open(args.datalist[:-4] + '_aug.txt').read().splitlines()
    img_ids = open(args.datalist[:-4] + '.txt').read().splitlines()
    
    # get arguments
    idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))
    
    mIOU1 = IOUMetric(num_classes=args.num_classes)
    mIOU2 = IOUMetric(num_classes=args.num_classes)
    st = time.time()

    miou_img_id1 = dict()
    miou_img_id2 = dict()
    diff_img_id = dict()
    for idx, img_id in tqdm(enumerate(img_ids)):
        mIOU1 = IOUMetric(num_classes=args.num_classes)
        mIOU2 = IOUMetric(num_classes=args.num_classes)
        
        gt_path = os.path.join(args.gt_dir, img_id + '.png')
        gt = Image.open(gt_path) # HW
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        
        pred_path1 = os.path.join(args.pred_dir1, prefix + img_id + subfix)
        pred1 = Image.open(pred_path1)
        pred1 = np.asarray(pred1)
        
        pred_path2 = os.path.join(args.pred_dir2, prefix + img_id + subfix)
        pred2 = Image.open(pred_path2)
        pred2 = np.asarray(pred2)
        
        mIOU1.add_batch(pred1, gt) # input to mIoU class
        mIOU2.add_batch(pred2, gt) # input to mIoU class
        acc1, recall1, precision1, TP1, TN1, FP1, cls_iu1, miou1, fwavacc1 = mIOU1.evaluate()
        acc2, recall2, precision2, TP2, TN2, FP2, cls_iu2, miou2, fwavacc2 = mIOU2.evaluate()
        miou_img_id1[img_id] = miou1
        miou_img_id2[img_id] = miou2
        diff_img_id[img_id] = abs(miou1 - miou2)
        
        del mIOU1, mIOU2

            
    diff_list = []
    img_id_list = []
    for key, value in sorted(diff_img_id.items(), key = lambda item: item[1], reverse=reverse): # low first ,,, reverse = True
        img_id_list.append(key)
        diff_list.append(value)
    diff_list = diff_list[:rank]
    img_id_list = img_id_list[:rank]

    print(diff_list)
    print(img_id_list)
        
    print("Difference of Top 5")
    print(diff_list[0])
    print(diff_list[1])
    print(diff_list[2])
    print(diff_list[3])
    print(diff_list[4])
    
    _work_label_2pred(args, img_id_list)
    
    
def _work_label_2pred(args, img_id_list):
    prefix = 'pseudo_mask_'
    subfix = '_1.png'
    # prefix = ''
    # subfix = '.png'

    img_ids = img_id_list

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        img.save(args.comparison_draw_dir +'/rank%d_%s_img.png' % (idx+1, img_id))
        w, h = img.size[0], img.size[1]

        pred_path1 = os.path.join(args.pred_dir1, prefix + img_id + subfix)
        cams1 = Image.open(pred_path1)
        cams1 = np.asarray(cams1)
        
        pred_path2 = os.path.join(args.pred_dir2, prefix + img_id + subfix)
        cams2 = Image.open(pred_path2)
        cams2 = np.asarray(cams2)
        
        cam_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[cams1])).convert("RGB") # cam_img_pil = VOC_COLORMAP[cams]
        # cam_img_pil.save(args.comparison_draw_dir +'/rank%d_%s_1_predict.png' % (idx+1, img_id))
        alphaComposited = Image.blend(img, cam_img_pil, 0.75)
        alphaComposited.save(args.comparison_draw_dir +'/rank%d_%s_1.png' % (idx+1, img_id))
        
        cam_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[cams2])).convert("RGB") # cam_img_pil = VOC_COLORMAP[cams]
        # cam_img_pil.save(args.comparison_draw_dir +'/rank%d_%s_2_predict.png' % (idx+1, img_id))
        alphaComposited = Image.blend(img, cam_img_pil, 0.75)
        alphaComposited.save(args.comparison_draw_dir +'/rank%d_%s_2.png' % (idx+1, img_id))
        
        gt_path = os.path.join(args.gt_dir, img_id + '.png')
        gt = Image.open(gt_path) # HW
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        gt[(gt==255)] = 0
        
        gt_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[gt])).convert("RGB") # gt_img_pil = VOC_COLORMAP[cams]
        # gt_img_pil.save(args.comparison_draw_dir +'/rank%d_%s_2_GTori.png' % (idx+1, img_id))
        alphaComposited = Image.blend(img, gt_img_pil, 0.75)
        alphaComposited.save(args.comparison_draw_dir +'/rank%d_%s_GT.png' % (idx + 1, img_id))
    
    print("Done!!!!!!!!!!! Lets Compare!!!!!!!!!!!!!!!!!!!")
    

##############################################

def evalNdrawN3comparison(args, rank=100, reverse=False):
    '''
    True :big val first
    '''
    
    # img_ids = open(args.datalist[:-4] + '_aug.txt').read().splitlines()
    img_ids = open(args.datalist[:-4] + '.txt').read().splitlines()
    
    # get arguments
    # idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))
    
    # mIOU1 = IOUMetric(num_classes=args.num_classes)
    # mIOU2 = IOUMetric(num_classes=args.num_classes)
    # mIOU3 = IOUMetric(num_classes=args.num_classes)
    st = time.time()

    miou_img_id1 = dict()
    miou_img_id2 = dict()
    miou_img_id3 = dict()
    diff_img_id = dict()
    for idx, img_id in tqdm(enumerate(img_ids)):
        mIOU1 = IOUMetric(num_classes=args.num_classes)
        mIOU2 = IOUMetric(num_classes=args.num_classes)
        mIOU3 = IOUMetric(num_classes=args.num_classes)
        
        gt_path = os.path.join(args.gt_dir, img_id + '.png')
        gt = Image.open(gt_path) # HW
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        
        pred_path1 = os.path.join(args.pred_dir1, img_id + '.png')
        pred1 = Image.open(pred_path1)
        pred1 = np.asarray(pred1)
        
        pred_path2 = os.path.join(args.pred_dir2, img_id + '.png')
        pred2 = Image.open(pred_path2)
        pred2 = np.asarray(pred2)
        
        pred_path3 = os.path.join(args.pred_dir3, img_id + '.png')
        pred3 = Image.open(pred_path3)
        pred3 = np.asarray(pred3)
        
        mIOU1.add_batch(pred1, gt) # input to mIoU class
        mIOU2.add_batch(pred2, gt) # input to mIoU class
        mIOU3.add_batch(pred3, gt) # input to mIoU class
        acc1, recall1, precision1, TP1, TN1, FP1, cls_iu1, miou1, fwavacc1 = mIOU1.evaluate()
        acc2, recall2, precision2, TP2, TN2, FP2, cls_iu2, miou2, fwavacc2 = mIOU2.evaluate()
        acc2, recall2, precision2, TP2, TN2, FP2, cls_iu2, miou3, fwavacc2 = mIOU3.evaluate()
        miou_img_id1[img_id] = miou1
        miou_img_id2[img_id] = miou2
        miou_img_id3[img_id] = miou3
        diff_img_id[img_id] = abs(miou1 - miou2) + abs(miou1 - miou3)
        
        del mIOU1, mIOU2

            
    diff_list = []
    img_id_list = []
    for key, value in sorted(diff_img_id.items(), key = lambda item: item[1], reverse=reverse): # low first ,,, reverse = True
        img_id_list.append(key)
        diff_list.append(value)
    diff_list = diff_list[:rank]
    img_id_list = img_id_list[:rank]

    print(diff_list)
    print(img_id_list)
        
    print("Difference of Top 5")
    print(diff_list[0])
    print(diff_list[1])
    print(diff_list[2])
    print(diff_list[3])
    print(diff_list[4])
    
    _work_label_3pred(args, img_id_list)
    
    
def _work_label_3pred(args, img_id_list):
    img_ids = img_id_list

    st = time.time()
    for idx, img_id in tqdm(enumerate(img_ids)):
        img_path = os.path.join(args.data_root, img_id + '.jpg')
        img = Image.open(img_path) # HW
        img.save(args.comparison_draw_dir +'/rank%d_%s_img.png' % (idx+1, img_id))
        w, h = img.size[0], img.size[1]

        pred_path1 = os.path.join(args.pred_dir1, img_id + '.png')
        cams1 = Image.open(pred_path1)
        cams1 = np.asarray(cams1)
        
        pred_path2 = os.path.join(args.pred_dir2, img_id + '.png')
        cams2 = Image.open(pred_path2)
        cams2 = np.asarray(cams2)
        
        pred_path3 = os.path.join(args.pred_dir3, img_id + '.png')
        cams3 = Image.open(pred_path3)
        cams3 = np.asarray(cams3)
        
        cam_img_pil1 = Image.fromarray(np.uint8(VOC_COLORMAP[cams1])).convert("RGB") # cam_img_pil1 = VOC_COLORMAP[cams]
        cam_img_pil1.save(args.comparison_draw_dir +'/rank%d_%s_1_predict.png' % (idx+1, img_id))
        
        cam_img_pil2 = Image.fromarray(np.uint8(VOC_COLORMAP[cams2])).convert("RGB") # cam_img_pil2 = VOC_COLORMAP[cams]
        cam_img_pil2.save(args.comparison_draw_dir +'/rank%d_%s_2_predict.png' % (idx+1, img_id))
        
        cam_img_pil3 = Image.fromarray(np.uint8(VOC_COLORMAP[cams3])).convert("RGB") # cam_img_pil3 = VOC_COLORMAP[cams]
        cam_img_pil3.save(args.comparison_draw_dir +'/rank%d_%s_3_predict.png' % (idx+1, img_id))
        
        gt_path = os.path.join(args.gt_dir, img_id + '.png')
        gt = Image.open(gt_path) # HW
        gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        gt[(gt==255)] = 0
        
        gt_img_pil = Image.fromarray(np.uint8(VOC_COLORMAP[gt])).convert("RGB") # gt_img_pil = VOC_COLORMAP[cams]
        gt_img_pil.save(args.comparison_draw_dir +'/rank%d_%s_GT.png' % (idx+1, img_id))
    
    print("Done!!!!!!!!!!! Lets Compare!!!!!!!!!!!!!!!!!!!")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", default='./dataset/VOCdevkit/VOC2012/JPEGImages', required=True, type=str) # 'dataset/VOCdevkit/VOC2012/JPEGImages'
    parser.add_argument("--dataset", default='voc12', type=str)
    parser.add_argument("--datalist", default="metadata/voc12/val.txt", type=str)
    parser.add_argument('--gt_dir', required=True, type=str)
    parser.add_argument('--pred_dir', required=True, type=str) # save npy cam
    parser.add_argument('--draw_dir', required=True, type=str)
    parser.add_argument('--pred_dir1', type=str) # save npy cam
    parser.add_argument('--pred_dir2', type=str) # save npy cam
    parser.add_argument('--pred_dir3', type=str) # save npy cam
    parser.add_argument('--comparison_draw_dir', type=str)
    
    args = parser.parse_args()
    _NUM_CLASSES = {'voc12': 20, 'coco': 80}
    args.num_classes = _NUM_CLASSES[args.dataset]
    
    os.makedirs(args.draw_dir, exist_ok=True)
    os.makedirs(args.comparison_draw_dir, exist_ok=True)
    timer = pyutils.Timer('module.make_orand_label:')
    
    # evalNdraw(args, rank=100, reverse=False)
    evalNdrawNcomparison(args, rank=300, reverse=True)
    # evalNdrawN3comparison(args, rank=200, reverse=True)


'''
git pull origin main && python3 miou_order.py \
    --data_root ./dataset/VOCdevkit/VOC2012/JPEGImages \
    --gt_dir ./dataset/VOCdevkit/VOC2012/SegmentationClassAug \
    --pred_dir ./segout/robust_amnmct_irn_val_ms_crf \
    --pred_dir1 ./deeplab/data/features/voc12_amn/deeplabv2_resnet101_msc/val/label_crf/ \
    --pred_dir2 ./segout/mctformer_val_ms_crf/ \
    --pred_dir3 ./segout/robust_amnmct_irn_val_ms_crf \
    --draw_dir ./finalstage_comparison/robust_amnmct_AMN/miou_order \
    --comparison_draw_dir ./finalstage_comparison/robust_amnmct/miou_order_comparison

git pull origin main && python3 miou_order.py \
    --data_root ./dataset/VOCdevkit/VOC2012/JPEGImages \
    --gt_dir ./dataset/VOCdevkit/VOC2012/SegmentationClassAug \
    --datalist metadata/voc12/train_aug.txt \
    --pred_dir ./dataset/PseudoMask/default/irn \
    --pred_dir1 ./dataset/PseudoMask/simpleaddition/irn \
    --pred_dir2 ./dataset/PseudoMask/default/irn \
    --draw_dir ./finalstage_comparison/def2simple/miou_order \
    --comparison_draw_dir ./finalstage_comparison/def2simple/miou_order_comparison
    
'''