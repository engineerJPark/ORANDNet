import random
import os.path
from PIL import Image
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as vision_tf
from util import imutils


IMG_FOLDER_NAME = "JPEGImages"
COCO_IMG_FOLDER_NAME = "train2014"
# IMG_FOLDER_NAME = "train2014"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

# cls_labels_dict = np.load('metadata/voc12/cls_labels.npy', allow_pickle=True).item()
# coco_cls_labels_dict = np.load('metadata/coco/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    # s = str(int(int_filename))

    s = str(int_filename).split('\n')[0]
    if len(s) != 12:
        s = '%012d' % int(s)
    return s

def cam2or(cam1, cam2):
    '''
    cam1, cam2 should be numpy tensor
    3 dimension
    '''
    outcam = cam1 + cam2 - cam1 * cam2
    return outcam

def cam2and(cam1, cam2):
    '''
    cam1, cam2 should be numpy tensor
    '''
    outcam = cam1 * cam2
    return outcam

def cam2xor(cam1, cam2):
    '''
    cam1, cam2 should be numpy tensor
    3 dimension
    '''
    outcam = cam1 + cam2 - cam1 * cam2 * 2
    return outcam

def pil2np_3dim(pilimage):
    """
    custom lambda
    """
    img = np.asarray(pilimage)
    if len(img.shape) == 3:    
        return np.asarray(img)
    elif len(img.shape) == 2:
        return np.asarray(np.stack((img, img, img), axis=-1))
    else:
        print("img.shape is ", img.shape)
        raise("img.shape")

def make_input_output(id, args):
    # get cam for resnet & vit
    # CAM has dimension as (n_classes, h, w)
    if 'orand' in args.network_type or args.network_type == 'encdec' or args.network_type == 'encdec_pred'\
        or args.network_type == 'crf' or args.network_type == 'irn':
        cam1 = np.load('./savefile/cam/result' + '/resnet/cam_npy/' + id + '_1.npy', allow_pickle=True).item()
        cam2 = np.load('./savefile/cam/result' + '/vit/cam_npy/'  + id + '_1.npy', allow_pickle=True).item()
        cam2['high_res'] = cam2['high_res'] + 0.15 ## compensating threshold difference
        cam2['cam'] = cam2['cam'] + 0.15 ## compensating threshold difference

        or_cam = cam2or(cam1['cam'], cam2['cam'])
        and_cam = cam2and(cam1['cam'], cam2['cam'])
        or_cam_highres = cam2or(cam1['high_res'], cam2['high_res'])
        and_cam_highres = cam2and(cam1['high_res'], cam2['high_res'])
        
        or_dict, and_dict = dict(), dict()
        or_dict['keys'], and_dict['keys'] = cam1['keys'], cam1['keys']
        or_dict['cam'], and_dict['cam'] = or_cam, and_cam
        or_dict['high_res'], and_dict['high_res'] = or_cam_highres, and_cam_highres

        # cam2['high_res'][cam2['high_res'] > 1] = 1.
        # cam2['cam'][cam2['cam'] > 1] = 1.
        
    elif args.network_type == 'encdec_amnmct' or args.network_type == 'encdec_pred_amnmct'\
        or args.network_type == 'crf_amnmct' or args.network_type == 'irn_amnmct' or 'orand_amnmct' in args.network_type:
        cam1 = np.load('./savefile/cam/result' + '/amn_cam/' + id + '.npy', allow_pickle=True).item()
        cam2 = np.load('./savefile/cam/result' + '/mctformer/'  + id + '.npy', allow_pickle=True).item()
        
        cam2list = [cam2[int(i)] for i in cam1['keys']]
        cam2 = np.stack(cam2list, axis=0)

        or_cam_highres = cam2or(cam1['high_res'][1:], cam2)
        and_cam_highres = cam2and(cam1['high_res'][1:], cam2)
        
        or_dict, and_dict = dict(), dict()
        or_dict['keys'], and_dict['keys'] = cam1['keys'], cam1['keys']
        or_dict['high_res'], and_dict['high_res'] = or_cam_highres, and_cam_highres
        
    elif args.network_type == 'encdec_coco' or args.network_type == 'encdec_pred_coco':
        cam1 = np.load('./savefile/coco/cam/result' + '/resnet/cam_npy/' + id + '_1.npy', allow_pickle=True).item()
        cam2 = np.load('./savefile/coco/cam/result' + '/vit/cam_npy/'  + id + '_1.npy', allow_pickle=True).item()
        cam2['high_res'] = cam2['high_res'] + 0.15 ## compensating threshold difference
        cam2['cam'] = cam2['cam'] + 0.15 ## compensating threshold difference

        or_cam = cam2or(cam1['cam'], cam2['cam'])
        and_cam = cam2and(cam1['cam'], cam2['cam'])
        or_cam_highres = cam2or(cam1['high_res'], cam2['high_res'])
        and_cam_highres = cam2and(cam1['high_res'], cam2['high_res'])
        
        or_dict, and_dict = dict(), dict()
        or_dict['keys'], and_dict['keys'] = cam1['keys'], cam1['keys']
        or_dict['cam'], and_dict['cam'] = or_cam, and_cam
        or_dict['high_res'], and_dict['high_res'] = or_cam_highres, and_cam_highres
        
    elif args.network_type == 'encdec_amnmct_coco' or args.network_type == 'encdec_pred_amnmct_coco':
        cam1 = np.load('./savefile/coco/cam/result' + '/amn_cam_coco/' + id.replace('COCO_train2014_', '') + '.npy', allow_pickle=True).item()
        cam2 = np.load('./savefile/coco/cam/result' + '/fused-patchrefine-npy/'  + id + '.npy', allow_pickle=True).item()
        
        cam2list = [cam2[int(i)] for i in cam1['keys']]
        cam2 = np.stack(cam2list, axis=0)

        or_cam_highres = cam2or(cam1['high_res'][1:], cam2)
        and_cam_highres = cam2and(cam1['high_res'][1:], cam2)
        
        or_dict, and_dict = dict(), dict()
        or_dict['keys'], and_dict['keys'] = cam1['keys'], cam1['keys']
        or_dict['high_res'], and_dict['high_res'] = np.float32(or_cam_highres), np.float32(and_cam_highres)

    else:
        raise NotImplementedError('no proper dataset!')

    return or_dict, and_dict # or_cam and and_cam label fg 0,1,2,3, ...

############################################################
############################################################
### for comparison test
def cam2simple_addition(cam1, cam2):
    '''
    cam1, cam2 should be numpy tensor
    3 dimension
    '''
    outcam = cam1 + cam2
    return outcam

### for comparison test
def make_simple_addition(id, args):
    # get cam for resnet & vit
    # CAM has dimension as (n_classes, h, w)

    cam1 = np.load('./savefile/cam/result' + '/resnet/cam_npy/' + id + '_1.npy', allow_pickle=True).item()
    cam2 = np.load('./savefile/cam/result' + '/vit/cam_npy/'  + id + '_1.npy', allow_pickle=True).item()
    cam2['high_res'] = cam2['high_res'] + 0.15 ## compensating threshold difference
    cam2['cam'] = cam2['cam'] + 0.15 ## compensating threshold difference

    add_cam = cam2simple_addition(cam1['cam'], cam2['cam'])
    add_cam_highres = cam2simple_addition(cam1['high_res'], cam2['high_res'])
    
    add_dict = dict()
    add_dict['keys'] = cam1['keys']
    add_dict['cam'] = add_cam
    add_dict['high_res'] = add_cam_highres

    return add_dict # or_cam and and_cam label fg 0,1,2,3, ...


def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
    
    # import pdb; pdb.set_trace()
    return [cls_labels_dict[img_name] for img_name in img_name_list]
    
    
class Multiscaling:
    '''
    multiscaling transformation function
    '''
    def __init__(self, 
                 transform=transforms.Compose([transforms.Lambda(pil2np_3dim), imutils.Normalize(), imutils.HWC_to_CHW]),
                 scale_list=(0.5, 1.0, 1.5, 2.0)):
        self.transform = transform
        self.scale_list = scale_list
        self.num_scales = len(scale_list)


    def __call__(self, image):
        '''
        image size : HWC
        '''
        self.multi_scale_image_list = list()
        self.multi_scale_flipped_image_list = list()
        
        img_size = image.size
        # insert multi-scale images
        for s in self.scale_list:
            target_size = (round(img_size[0] * s), round(img_size[1] * s))
            scaled_image = image.resize(target_size, resample=Image.BILINEAR) # CUBIC
            self.multi_scale_image_list.append(scaled_image) # contain 4 images
            
        # transform the multi-scaled image
        for i in range(self.num_scales):
            self.multi_scale_image_list[i] = self.transform(self.multi_scale_image_list[i]) # contain 4 images, CHW

        # augment the flipped image
        for i in range(self.num_scales):
            self.multi_scale_flipped_image_list.append(
                np.stack(
                    [self.multi_scale_image_list[i], np.flip(self.multi_scale_image_list[i], -1).copy()]
                    ,axis=0
                ) # 2CHW -> 4 2CHW
            )
            
        # print('end')
        return self.multi_scale_flipped_image_list

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB") # Width, Height
        ori_size = (img.size[1], img.size[0]) # Height Width 
        
        if self.transform:
            img = self.transform(img)
        
        return img_id, img, ori_size


class ClassificationDataset(ImageDataset):
    """
    Classification Dataset (base)
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        super().__init__(dataset, img_id_list_file, img_root, transform)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)
        self.from_numpy = torch.from_numpy

    def __getitem__(self, idx):
        name, img, ori_size = super().__getitem__(idx)
        label = self.from_numpy(self.label_list[idx])
        return {'name':name, 'img':img, 'label':label, 'size':ori_size} # img can be list in inference time. batch 1


class ClassificationDataset_MultiScale(ClassificationDataset):
    """
    Classification Dataset (base)
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=Multiscaling()):
        super().__init__(dataset, img_id_list_file, img_root, transform)
        

    def __getitem__(self, idx):
        pack = super().__getitem__(idx)

        return {"name": pack['name'], "img": pack['img'], # img is list
                "size": pack['size'],
               "label": pack['label']}
        

class Or2AndCamDataset(Dataset):
    def __init__(self, dataset, img_id_list_file, img_root, args, rescale=None, crop_size=None, network_type=None, training=False):
        self.img_id_list = load_img_id_list(img_id_list_file) # args.datalist
        self.img_root = img_root
        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)
        self.from_numpy = torch.from_numpy
        self.make_input_output = make_input_output
        self.args = args
        
        self.training = training
        self.random_scale = imutils.RandomResizeLong_numpy(args.resize_size[0], args.resize_size[1])
        
        self.transform=transforms.Compose([
                    np.asarray,
                    imutils.Normalize(),
                    imutils.HWC_to_CHW,
                    torch.from_numpy])
        
    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB") # Width, Height
        ori_size = (img.size[1], img.size[0]) # Height Width 
        label = self.from_numpy(self.label_list[idx])
        input_dict, label_dict = self.make_input_output(img_id, self.args) # [CHW]
        
        img = self.transform(img)
        
        # wrapped by batch_size b=1
        return {'name':img_id, 'img':img, 'input_cam_highres':self.from_numpy(input_dict['high_res']), 
                'label_cam_highres':self.from_numpy(label_dict['high_res']), 'label':label, 'size':ori_size} # numpt array 
        
#################################################for IRN######################################################
#################################################for IRN######################################################
#################################################for IRN######################################################

import imageio

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

# cls_labels_dict = np.load('metadata/voc12/cls_labels.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    # s = str(int(int_filename))

    s = str(int_filename).split('\n')[0]
    if len(s) != 12:
        s = '%012d' % int(s)
    return s

def get_img_path(img_name, voc12_root, dataset=None):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    if dataset == 'coco':
        return os.path.join(voc12_root, COCO_IMG_FOLDER_NAME, 'COCO_train2014_' + img_name + '.jpg')
    else:
        return os.path.join(voc12_root, img_name + '.jpg')

# def load_image_label_list_from_npy(img_name_list, dataset=None): 
#     return np.array([cls_labels_dict[int(img_name)] for img_name in img_name_list])
    # if dataset == 'coco':
    #     return np.array([coco_cls_labels_dict[img_name] for img_name in img_name_list])
    # else:
    #     return np.array([cls_labels_dict[int(img_name)] for img_name in img_name_list])

class VOC12SegmentationDataset(Dataset):
    
    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root, 
                 rescale=None, img_normal=imutils.Normalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_id_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.label_dir = label_dir

        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name_str = self.img_name_list[idx] # name_str = decode_int_filename(name)
        

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        label = imageio.imread(os.path.join(self.label_dir, name_str + '.png'))

        img = np.asarray(img)
        if len(img.shape)==2:
            img = np.stack((img, img, img), axis=-1)

        if self.rescale:
            img, label = imutils.random_scale((img, label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, label = imutils.random_lr_flip((img, label))

        if self.crop_method == "random":
            img, label = imutils.random_crop((img, label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            label = imutils.top_left_crop(label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)
        
        return {'name': name_str, 'img': img, 'label': label}
        # return {'name': name_str, 'img_list': img_list, 'label_list': label_list}
        
        
class VOC12SegmentationDataset4IRN(Dataset):
    
    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 rescale=None, img_normal=imutils.Normalize(), hor_flip=False,
                 crop_method = 'random'):

        self.img_name_list = load_img_id_list(img_name_list_path)
        self.pseudo_label_list = [st + '_1' for st in self.img_name_list]
        
        # self.pseudo_label_list.extend([st + '_2' for st in self.img_name_list])
        # self.pseudo_label_list.extend([st + '_3' for st in self.img_name_list])
        # self.pseudo_label_list.extend([st + '_4' for st in self.img_name_list])
        # self.img_name_list = self.img_name_list * 4
        
        self.voc12_root = voc12_root
        self.label_dir = label_dir
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name_str = self.img_name_list[idx] # name_str = decode_int_filename(name)
        pseudo_name_str = self.pseudo_label_list[idx]

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        pseudo_label = imageio.imread(os.path.join(self.label_dir, pseudo_name_str + '.png'))
        
        img = np.asarray(img)
        if len(img.shape)==2:
            img = np.stack((img, img, img), axis=-1)
            
        # print(np.unique(img))
        # print(np.unique(pseudo_label))

        if self.rescale:
            img, pseudo_label = imutils.random_scale((img, pseudo_label), scale_range=self.rescale, order=(3, 0))

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img, pseudo_label = imutils.random_lr_flip((img, pseudo_label))

        if self.crop_method == "random":
            img, pseudo_label = imutils.random_crop((img, pseudo_label), self.crop_size, (0, 255))
        else:
            img = imutils.top_left_crop(img, self.crop_size, 0)
            pseudo_label = imutils.top_left_crop(pseudo_label, self.crop_size, 255)

        img = imutils.HWC_to_CHW(img)
        
        return {'name': name_str, 'img': img, 'label': pseudo_label}
    
            
class VOC12AffinityDataset(VOC12SegmentationDataset4IRN):
    def __init__(self, img_name_list_path, label_dir, crop_size, voc12_root,
                 indices_from, indices_to,
                 rescale=None, img_normal=imutils.Normalize(), hor_flip=False, crop_method=None):
        super().__init__(img_name_list_path, label_dir, crop_size, voc12_root, rescale, img_normal, hor_flip, crop_method=crop_method)

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)

        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced_label)

        return out

class GetAffinityLabelFromIndices():
    
    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

    
class VOC12ImageDataset(Dataset):
    
    def __init__(self, img_name_list_path, voc12_root,
                 resize_long=None, rescale=None, img_normal=imutils.Normalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_id_list(img_name_list_path)
        self.voc12_root = voc12_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name_str = self.img_name_list[idx] # name_str = decode_int_filename(name)
        
        img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))
        if len(img.shape)==2:
            img = np.stack((img, img, img), axis=-1)

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)
        
        return {'name': name_str, 'img': img}
    
class VOC12ClassificationDataset(VOC12ImageDataset):
    
    def __init__(self, img_name_list_path, dataset, voc12_root,
                 resize_long=None, rescale=None, img_normal=imutils.Normalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, voc12_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_img_label_list_from_npy(self.img_name_list, dataset)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out

class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, dataset, voc12_root,
                 img_normal=imutils.Normalize(),
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, dataset, voc12_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name_str = self.img_name_list[idx] # name_str = decode_int_filename(name)

        img = imageio.imread(get_img_path(name_str, self.voc12_root))
        if len(img.shape)==2:
            img = np.stack((img, img, img), axis=-1)

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]
        
        # print("img.shape: ", img.shape)
        
        out = {"name": name_str, "img": ms_img_list, "size": (img.shape[0], img.shape[1]), "label": torch.from_numpy(self.label_list[idx])}
        return out