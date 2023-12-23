import PIL.Image
import random
import cv2

import torch
import numpy as np


# def make_input_output_highres(id, args):
#     # get cam for resnet & vit
#     # CAM has dimension as (n_classes, h, w)
#     if args.network_type == 'encdec' or args.network_type == 'encdec_pred'\
#         or args.network_type == 'crf':
#         cam1 = np.load('savefile/cam/result' + '/resnet/cam_npy/' + id + '.npy', allow_pickle=True).item()['high_res']
#         cam2 = np.load('savefile/cam/result' + '/vit/cam_npy/'  + id + '.npy', allow_pickle=True).item()['high_res']
#     elif args.network_type == 'encdec_cgn' or args.network_type == 'encdec_pred_cgn':
#         cam1 = np.load('savefile/cam/result' + '/resnet_cgn/cam_npy/' + id + '.npy', allow_pickle=True).item()['high_res']
#         cam2 = np.load('savefile/cam/result' + '/vit_cgn/cam_npy/'  + id + '.npy', allow_pickle=True).item()['high_res'] 
#     else:
#         raise NotImplementedError('no proper dataset!')
    
#     # get input, output
#     ivr_normalization = True if 'ivr' in args.network_type else False
    
#     or_cam = cam2or(cam1, cam2, ivr_normalization=ivr_normalization)
#     and_cam = cam2and(cam1, cam2, ivr_normalization=ivr_normalization)
#     return or_cam, and_cam # label fg 0,1,2,3, ...

def ivr_normalization(tensor, percentile=0.4):
    '''
    tensor is 3 dimension, CHW
    '''
    if isinstance(tensor, np.ndarray):
        percentile_val = np.amin(tensor, axis=(-2,-1), keepdims=True) + \
            (np.amax(tensor, axis=(-2,-1), keepdims=True) - np.amin(tensor, axis=(-2,-1), keepdims=True)) * percentile
        tensor[tensor < percentile_val] = 0.
        tensor /= np.amax(tensor, axis=(-2,-1), keepdims=True) + 1e-5
        # numer = tensor - percentile_val
        # denom = np.amax(tensor - percentile_val, axis=(-2,-1), keepdims=True) + 1e-5
        return tensor # (chw)
    
    elif isinstance(tensor, torch.Tensor):
        percentile_val = torch.amin(tensor, dim=(-2,-1), keepdim=True) + \
            (torch.amax(tensor, dim=(-2,-1), keepdim=True) - torch.amin(tensor, dim=(-2,-1), keepdim=True)) * percentile
        tensor[tensor < percentile_val] = 0.
        tensor /= torch.amax(tensor, dim=(-2,-1), keepdim=True) + 1e-5
        # numer = tensor - percentile_val
        # denom = torch.amax(tensor - percentile_val, dim=(-2,-1), keepdim=True) + 1e-5
        return tensor # (chw)
    
    else:
        raise NotImplementedError("No Class type is in here")

class RandomResizeLong:

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, target_long=None, mode='image'):
        if target_long is None:
            target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        if mode == 'image':
            img = img.resize(target_shape, resample=PIL.Image.CUBIC)
        elif mode == 'mask':
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)

        return img
    
class RandomResizeLong_numpy:
    
    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, target_long=None):
        if target_long is None:
            target_long = random.randint(self.min_long, self.max_long)
        h, w = img[0].shape[:2]

        if w < h:
            target_shape = (target_long, int(round(w * target_long / h)))
        else:
            target_shape = (int(round(h * target_long / w)), target_long)
            
        img_list = list()
        for i in range(len(img)):
            # print(i)
            
            
            if i == 0:
                img_sub = cv2.resize(img[i], dsize=target_shape, interpolation=cv2.INTER_CUBIC)
            else:
                img_sub = cv2.resize(img[i], dsize=target_shape, interpolation=cv2.INTER_NEAREST)
            
            img_list.append(img_sub)
        return img_list


class RandomCrop:

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0

        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]

        return container

class RandomHorizontalFlip():
    def __init__(self):
        return

    def __call__(self, img):
        if bool(random.getrandbits(1)):
            img = np.fliplr(img).copy()
        return img


class CenterCrop():

    def __init__(self, cropsize, default_value=0):
        self.cropsize = cropsize
        self.default_value = default_value

    def __call__(self, npimg):

        h, w = npimg.shape[:2]

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)

        sh = h - self.cropsize
        sw = w - self.cropsize

        if sw > 0:
            cont_left = 0
            img_left = int(round(sw / 2))
        else:
            cont_left = int(round(-sw / 2))
            img_left = 0

        if sh > 0:
            cont_top = 0
            img_top = int(round(sh / 2))
        else:
            cont_top = int(round(-sh / 2))
            img_top = 0

        if len(npimg.shape) == 2:
            container = np.ones((self.cropsize, self.cropsize), npimg.dtype)*self.default_value
        else:
            container = np.ones((self.cropsize, self.cropsize, npimg.shape[2]), npimg.dtype)*self.default_value

        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            npimg[img_top:img_top+ch, img_left:img_left+cw]

        return container


def HWC_to_CHW(img):
    # print(img.shape)
    return np.transpose(img, (2, 0, 1))

###############################################################
###############################################################


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=-1)
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)

###############################################################
###############################################################

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=-1)
    h, w = img.shape[:2] # img dimension hwc
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def _crf_with_alpha(image, cam_dict, alpha, t=10):
    # v = np.array(list(cam_dict.values()))
    v = cam_dict['high_res']
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha) # adding background
    bgcam_score = np.concatenate((bg_score, v), axis=0) # adding background
    crf_score = crf_inference(image, bgcam_score, labels=bgcam_score.shape[0], t=t) # (n_labels, h, w)
    
    cam_dict['high_res'] = crf_score # crf_score[1:] # (n_labels, h, w)
    return cam_dict, crf_score # w/ background

    # n_crf_al = dict()
    # n_crf_al[0] = crf_score[0]
    # for i, key in enumerate(cam_dict['keys']):
    #     cam_dict['high_res'] = crf_score[i+1] # n_crf_al[key+1] = crf_score[i+1]

###############################################################
###############################################################

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_arr = np.asarray(img)
        normalized_img = np.empty_like(img_arr, np.float32)

        normalized_img[..., 0] = (img_arr[..., 0] / 255. - self.mean[0]) / self.std[0]
        normalized_img[..., 1] = (img_arr[..., 1] / 255. - self.mean[1]) / self.std[1]
        normalized_img[..., 2] = (img_arr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return normalized_img


def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)


def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

#######################New Utils#######################
#######################for IRN#######################

from PIL import Image

def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
        
    # print(np.unique(img))

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def random_resize_long(img, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = img.shape[:2]

    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w

    return pil_rescale(img, scale, 3)

def random_scale(img, scale_range, order):

    target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

    if isinstance(img, tuple):
        return (pil_rescale(img[0], target_scale, order[0]), pil_rescale(img[1], target_scale, order[1]))
    else:
        return pil_rescale(img[0], target_scale, order)

def random_lr_flip(img):

    if bool(random.getrandbits(1)):
        if isinstance(img, tuple):
            return [np.fliplr(m) for m in img]
        else:
            return np.fliplr(img)
    else:
        return img

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def random_crop(images, cropsize, default_values):

    if isinstance(images, np.ndarray): images = (images,)
    if isinstance(default_values, int): default_values = (default_values,)

    imgsize = images[0].shape[:2]
    box = get_random_crop_box(imgsize, cropsize)

    new_images = []
    for img, f in zip(images, default_values):

        if len(img.shape) == 3:
            cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*f
        else:
            cont = np.ones((cropsize, cropsize), img.dtype)*f
        cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]
        new_images.append(cont)

    if len(new_images) == 1:
        new_images = new_images[0]

    return new_images

def top_left_crop(img, cropsize, default_value):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[:ch, :cw] = img[:ch, :cw]

    return container

def center_crop(img, cropsize, default_value=0):

    h, w = img.shape[:2]

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    sh = h - cropsize
    sw = w - cropsize

    if sw > 0:
        cont_left = 0
        img_left = int(round(sw / 2))
    else:
        cont_left = int(round(-sw / 2))
        img_left = 0

    if sh > 0:
        cont_top = 0
        img_top = int(round(sh / 2))
    else:
        cont_top = int(round(-sh / 2))
        img_top = 0

    if len(img.shape) == 2:
        container = np.ones((cropsize, cropsize), img.dtype)*default_value
    else:
        container = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value

    container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        img[img_top:img_top+ch, img_left:img_left+cw]

    return container

# def HWC_to_CHW(img):
#     return np.transpose(img, (2, 0, 1))

# def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
#     import pydensecrf.densecrf as dcrf
#     from pydensecrf.utils import unary_from_labels
#     h, w = img.shape[:2]

#     d = dcrf.DenseCRF2D(w, h, n_labels)

#     unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

#     d.setUnaryEnergy(unary)
#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

#     q = d.inference(t)

#     return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def compress_range(arr):
    uniques = np.unique(arr)
    maximum = np.max(uniques)

    d = np.zeros(maximum+1, np.int32)
    d[uniques] = np.arange(uniques.shape[0])

    out = d[arr]
    return out - np.min(out)