import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing
from PIL import Image
from tqdm import tqdm
from util import torchutils


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


# if __name__ == '__main__':
def eval_irn(args):
    
    # if True:
    #     thres_shift(args)
        
    # get arguments
    idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))

    img_ids = open(args.datalist).read().splitlines()
    st = time.time()
    post = ['_1']

    best_miou = 0
    best_prec = 0
    best_thres = 0
    # for thres in range(int(args.thres), 91, 1):
        # thres = thres * 0.01
    thres = args.thres
    thres_shift(args.thres, args)
        
    mIOU = IOUMetric(num_classes=args.num_classes)
    for i in range(len(post)):
        for idx, img_id in tqdm(enumerate(img_ids)):
            gt_path = os.path.join(args.gt_dir, img_id + '.png')
            gt = Image.open(gt_path) # HW
            w, h = gt.size[0], gt.size[1]
            gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
            
            pred_path = os.path.join(args.sem_seg_out_dir, 'pseudo_mask_%s%s.png' % (img_id, post[i]))
            pred = Image.open(pred_path) # pred = pred.crop((0, 0, w, h))
            pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
            
            mIOU.add_batch(pred, gt)
            
        acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()

        mean_prec = np.nanmean(precision)
        mean_recall = np.nanmean(recall)
        

        print(acc)
        with open(os.path.join(args.log_root, args.network_type + '_metric.log'), 'w') as f:
            f.write("{:>5} {:>20} {:>10} {:>10} {:>10}\n".format('IDX', 'Name', 'IoU', 'Prec', 'Recall'))
            f.write("{:>5} {:>20} {:>10.2f} {:>10.2f} {:>10.2f}\n".format(
                '-', 'mean', miou * 100, mean_prec * 100, mean_recall * 100))
            for i in range(args.num_classes):
                f.write("{:>5} {:>20} {:>10.2f} {:>10.2f} {:>10.2f}\n".format(
                    idx2num[i], idx2label[i][:10], cls_iu[i] * 100, precision[i] * 100, recall[i] * 100))
        print("{:>8} {:>8} {:>8} {:>8} {:>8}".format('IDX', 'IoU', 'Prec', 'Recall', 'ACC'))
        print("{:>8} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
            'mean', miou * 100, mean_prec * 100, mean_recall * 100, np.mean(acc) * 100))

        if mean_prec > mean_recall:
            best_miou = miou
            best_prec = precision
            best_thres = thres
            print('best threshold = ', thres)
            return
            
        result = {"Recall": ["{:.2f}".format(i) for i in recall.tolist()],
                "Precision": ["{:.2f}".format(i) for i in precision.tolist()],
                "Mean_Recall": mean_recall,
                "Mean_Precision": mean_prec,
                "IoU": cls_iu,
                "Mean IoU": miou,
                "TP": TP.tolist(),
                "TN": TN.tolist(),
                "FP": FP.tolist()}
        

def _work_thres_shift(process_id, img_ids_set, thres, args):
    def load_img_label_list_from_npy(img_name_list, dataset):
        cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
        return [cls_labels_dict[img_name] for img_name in img_name_list]
    
    img_ids = img_ids_set[process_id]
    for idx, img_id in tqdm(enumerate(img_ids)):
        if args.dataset == 'coco':
            label = load_img_label_list_from_npy([img_id], 'coco')[0]
        else:
            label = load_img_label_list_from_npy([img_id], 'voc12')[0]
        valid_cat = np.nonzero(label)[0]
        
        # pred_path = os.path.join(args.pred_dir, img_id + '%s.npy'%(post[i]))
        pred_path = os.path.join(args.pred_dir, img_id + '_1.npy')
        cams = np.load(pred_path, allow_pickle=True) # (2, 281, 500) (2, 384, 384)
        keys = np.pad(valid_cat + 1, (1, 0), mode='constant')
        # cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thres)
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres)
        cls_labels = np.argmax(cams, axis=0)
        pred = keys[cls_labels]
        pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]

        # for saving
        pred = Image.fromarray(pred)
        pred.save(args.sem_seg_out_dir + '/pseudo_mask_%s_1.png' % (img_id))

        
def thres_shift(thres, args):
    def load_img_label_list_from_npy(img_name_list, dataset):
        cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
        return [cls_labels_dict[img_name] for img_name in img_name_list]
    
    # best_thres = finding_best_thres_prec(args)
    # best_thres = 0.7

    st = time.time()
    if args.dataset != 'coco':
        img_ids = open(args.datalist[:-4] + '_aug.txt').read().splitlines()
    else:
        img_ids = open(args.datalist).read().splitlines()
    img_ids = torchutils.split_dataset(img_ids, args.num_workers)
    multiprocessing.spawn(_work_thres_shift, nprocs=args.num_workers, args=(img_ids, thres, args), join=True)
    

def finding_best_thres_prec(args):
    def load_img_label_list_from_npy(img_name_list, dataset):
        cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
        return [cls_labels_dict[img_name] for img_name in img_name_list]

    # get arguments
    idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))

    st = time.time()
    img_ids = open(args.datalist).read().splitlines()
    
    
    best_miou = 0
    best_thres = 0
    for thres_ in range(40, 80 + 1):
        thres = thres_ * 0.01
        mIOU = IOUMetric(num_classes=args.num_classes)
        for idx, img_id in tqdm(enumerate(img_ids)):
            if args.dataset == 'coco':
                label = load_img_label_list_from_npy([img_id], 'coco')[0]
            else:
                label = load_img_label_list_from_npy([img_id], 'voc12')[0]
            valid_cat = np.nonzero(label)[0]
            
            gt_path = os.path.join(args.gt_dir, img_id + '.png')
            gt = Image.open(gt_path) # HW
            w, h = gt.size[0], gt.size[1]
            gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary

            pred_path = os.path.join(args.pred_dir, img_id + '_1.npy')
            cams = np.load(pred_path, allow_pickle=True) # (2, 281, 500) (2, 384, 384)
            
            # import pdb; pdb.set_trace()
            # cams = torch.from_numpy(cam_dict['high_res']) # (2, 281, 500) (2, 384, 384)
            
            keys = np.pad(valid_cat + 1, (1, 0), mode='constant')
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres)
            cls_labels = np.argmax(cams, axis=0)
            
            pred = keys[cls_labels]
            pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
            
            mIOU.add_batch(pred, gt)

        acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()
        mean_prec = np.nanmean(precision)
        mean_recall = np.nanmean(recall)
        
        
        if mean_prec > mean_recall:
            best_thres = thres
            print("{:>8} {:>8} {:>8} {:>8} {:>8}".format('IDX', 'IoU', 'Prec', 'Recall', 'ACC'))
            print("{:>8} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
                'mean', miou * 100, mean_prec * 100, mean_recall * 100, np.mean(acc) * 100))
            break
        else:
            best_miou = miou
            best_thres = thres
            del mIOU
            print("current threshold:", thres)
            print("{:>8} {:>8} {:>8} {:>8} {:>8}".format('IDX', 'IoU', 'Prec', 'Recall', 'ACC'))
            print("{:>8} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}".format(
                'mean', miou * 100, mean_prec * 100, mean_recall * 100, np.mean(acc) * 100))
    
    print("============================================")
    print("best threshold:", best_thres)

    return best_thres
    # img_ids = torchutils.split_dataset(img_ids, args.num_workers)
    # multiprocessing.spawn(_work_finding_best_thres_prec, nprocs=args.num_workers, args=(img_ids, args), join=True)