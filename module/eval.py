import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


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
def eval(args):
    
    print("strict thres")
    best_thres = args.thres

    # img_ids = open(args.datalist[:-4] + '_aug.txt').read().splitlines()
    img_ids = open(args.datalist[:-4] + '.txt').read().splitlines()
    
    # get arguments
    idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))

    # mIOU = IOUMetric(num_classes=args.num_classes)

    st = time.time()

    post = ["_1"]
    
    mIOU_list = [IOUMetric(num_classes=args.num_classes) for _ in range(len(post))]
    for i in range(len(post)):
        for idx, img_id in tqdm(enumerate(img_ids)):
            gt_path = os.path.join(args.gt_dir, img_id + '.png')
            gt = Image.open(gt_path) # HW
            w, h = gt.size[0], gt.size[1]
            gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary
            
            if args.network_type == 'pseudo':
                pred_path = os.path.join(args.pred_dir, 'pseudo_mask_' + img_id + '%s.png'%(post[i]))
                pred = Image.open(pred_path)
                pred = np.asarray(pred)
                
            elif args.network_type == 'resnet50_cam' or args.network_type == 'vit_cam':
                pred_path = os.path.join(args.pred_dir, img_id + '%s.npy'%(post[i]))
                cam_dict = np.load(pred_path, allow_pickle=True).item() # (2, 281, 500) (2, 384, 384)
                cams = torch.from_numpy(cam_dict['high_res']) # (2, 281, 500) (2, 384, 384)
                
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.thres)
                cls_labels = np.argmax(cams, axis=0)
                pred = keys[cls_labels]
                pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]

            elif args.network_type == 'amn':
                # pred = Image.open(pred_path)
                pred_path = os.path.join(args.pred_dir, img_id + '.npy')\
                    
                # import pdb; pdb.set_trace()
                cam_dict = np.load(pred_path, allow_pickle=True).item() # (2, 281, 500) (2, 384, 384)
                cams = cam_dict['high_res'][1:, ...]# (2, 281, 500) (2, 384, 384)
                
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thres)

                cls_labels = np.argmax(cams, axis=0)
                pred = keys[cls_labels]
                pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]

            elif args.network_type == 'mctformer':
                # pred = Image.open(pred_path)
                pred_path = os.path.join(args.pred_dir, img_id + '.npy')
                    
                cam_dict = np.load(pred_path, allow_pickle=True).item() # (2, 281, 500) (2, 384, 384)
                
                # import pdb; pdb.set_trace()
                cams = torch.from_numpy(np.stack(list(cam_dict.values()), axis=0)) # (2, 281, 500) (2, 384, 384)
                keys = np.pad(np.array(list(cam_dict.keys())) + 1, (1, 0), mode='constant')
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thres)
                cls_labels = np.argmax(cams, axis=0)
                pred = keys[cls_labels]
                pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
                
            elif 'simpleaddition' in args.network_type:
                # pred = Image.open(pred_path)
                pred_path = os.path.join(args.pred_dir, img_id + '%s.npy'%(post[i]))
                    
                import pdb; pdb.set_trace()
                cam_dict = np.load(pred_path, allow_pickle=True) # (2, 281, 500) (2, 384, 384)
                cams = torch.from_numpy(cam_dict['high_res']) # (2, 281, 500) (2, 384, 384)
                
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thres)
                cls_labels = np.argmax(cams, axis=0)
                pred = keys[cls_labels]
                pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
                
            else:
                # pred = Image.open(pred_path)
                pred_path = os.path.join(args.pred_dir, img_id + '%s.npy'%(post[i]))\
                    
                # import pdb; pdb.set_trace()
                cam_dict = np.load(pred_path, allow_pickle=True).item() # (2, 281, 500) (2, 384, 384)
                cams = torch.from_numpy(cam_dict['high_res']) # (2, 281, 500) (2, 384, 384)
                
                keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=best_thres)
                cls_labels = np.argmax(cams, axis=0)
                pred = keys[cls_labels]
                pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
            
            mIOU_list[i].add_batch(pred, gt) # input to mIoU class

            # # for saving
            # pred = Image.fromarray(pred)
            # pred.save(args.sem_seg_out_dir + '/pseudo_mask_%s%s.png' % (img_id, post[i]))
        
        acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU_list[i].evaluate()
        
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


        result = {"Recall": ["{:.2f}".format(i) for i in recall.tolist()],
                "Precision": ["{:.2f}".format(i) for i in precision.tolist()],
                "Mean_Recall": mean_recall,
                "Mean_Precision": mean_prec,
                "IoU": cls_iu,
                "Mean IoU": miou,
                "TP": TP.tolist(),
                "TN": TN.tolist(),
                "FP": FP.tolist()}


def finding_best_thres(args):
    # get arguments
    idx2num, idx2label = get_labels(os.path.join('metadata', args.dataset, 'labels.txt'))

    img_ids = open(args.datalist).read().splitlines()
    st = time.time()
    
    best_miou = 0
    best_thres = 0
    count = 0
    for thres_ in range(10, 80 + 1):
        thres = thres_ * 0.01
        mIOU = IOUMetric(num_classes=args.num_classes)
        for idx, img_id in tqdm(enumerate(img_ids)):
            gt_path = os.path.join(args.gt_dir, img_id + '.png')
            gt = Image.open(gt_path) # HW
            w, h = gt.size[0], gt.size[1]
            gt = np.array(gt, dtype=np.uint8)  # shape = [h, w], 0-20 is classes, 255 is ingore boundary

            pred_path = os.path.join(args.pred_dir, img_id + '_1.npy')
            cam_dict = np.load(pred_path, allow_pickle=True).item() # (2, 281, 500) (2, 384, 384)
            cams = torch.from_numpy(cam_dict['high_res']) # (2, 281, 500) (2, 384, 384)

                
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=thres)
            cls_labels = np.argmax(cams, axis=0)
            
            pred = keys[cls_labels]
            pred = np.array(pred, dtype=np.uint8)  # shape = [h, w]
            
            mIOU.add_batch(pred, gt)

        acc, recall, precision, TP, TN, FP, cls_iu, miou, fwavacc = mIOU.evaluate()
        mean_prec = np.nanmean(precision)
        mean_recall = np.nanmean(recall)
        
        if mean_prec > mean_recall:
        # if best_miou > miou:
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