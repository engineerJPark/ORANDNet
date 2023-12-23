import matplotlib.pyplot as plt
from PIL import Image
import os

def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()

def get_concat_h(im1, im2, im3):
    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    return dst

os.makedirs('savefile/cam/result/comparison/cam_draw/', exist_ok=True)

im1_list = os.listdir('savefile/cam/result/resnet/cam_draw/')
im2_list = os.listdir('savefile/cam/result/vit/cam_draw/')
im3_list = os.listdir('savefile/cam/result/encdec/cam_draw/')

im1_list.sort()
im2_list.sort()
im3_list.sort()


for i in range(len(im1_list)):
    
    im1 = Image.open('savefile/cam/result/resnet/cam_draw/' + im1_list[i]).convert('RGB')
    im2 = Image.open('savefile/cam/result/vit/cam_draw/' + im2_list[i]).convert('RGB')
    im3 = Image.open('savefile/cam/result/encdec/cam_draw/' + im3_list[i]).convert('RGB')

    get_concat_h(im1, im2, im3).save('savefile/cam/result/comparison/cam_draw/' + im1_list[i])

