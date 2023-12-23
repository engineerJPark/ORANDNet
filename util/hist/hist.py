import numpy as np
import matplotlib.pyplot as plt
import os

def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()

os.makedirs('savefile/hist/plain/', exist_ok=True)

im1_list = os.listdir('savefile/cam/result/resnet/cam_npy/')
im2_list = os.listdir('savefile/cam/result/vit/cam_npy/')
im3_list = os.listdir('savefile/cam/result/encdec/cam_npy/')

temp1 = list() # np.zeros((1000, ), dtype=np.float32)
temp2 = list() # np.zeros((1000, ), dtype=np.float32)
temp3 = list() # np.zeros((1000, ), dtype=np.float32)

# print(len(im1_list))

for i in range(len(im1_list)):    
    cam1 = np.load(os.path.join('savefile/cam/result/resnet/cam_npy/', im1_list[i]), allow_pickle=True).item()['high_res']
    cam2 = np.load(os.path.join('savefile/cam/result/vit/cam_npy/', im2_list[i]), allow_pickle=True).item()['high_res']
    cam3 = np.load(os.path.join('savefile/cam/result/encdec/cam_npy/', im3_list[i]), allow_pickle=True).item()['high_res']

    temp1.extend(np.histogram(cam1, bins=1000, range=(0, 1), density=True)[0].tolist())
    temp2.extend(np.histogram(cam2, bins=1000, range=(0, 1), density=True)[0].tolist())
    temp3.extend(np.histogram(cam3, bins=1000, range=(0, 1), density=True)[0].tolist())

    # print(np.min(cam1), np.max(cam1))
    # print(np.min(cam2), np.max(cam2))
    # print(np.min(cam3), np.max(cam3))
    
    # temp1 += np.histogram(cam1, bins=1000, range=(0, 1), density=True)[0]
    # temp2 += np.histogram(cam2, bins=1000, range=(0, 1), density=True)[0]
    # temp3 += np.histogram(cam3, bins=1000, range=(0, 1), density=True)[0]
    
# print(np.min(temp1), np.max(temp1))
# print(np.min(temp2), np.max(temp2))
# print(np.min(temp3), np.max(temp3))

plt.figure()
plt.xlim(0,1)
_ = plt.hist(temp1, bins=10000)
plt.savefig('savefile/hist/plain/' +'/histogram_resnet.png')
plt.clf()
# np.savetxt('savefile/hist/plain/temp1.txt', temp1, delimiter = '\n')  

plt.figure()
plt.xlim(0,1)
_ = plt.hist(temp2, bins=10000)
plt.savefig('savefile/hist/plain/' +'/histogram_vit.png')
plt.clf()
# np.savetxt('savefile/hist/plain/temp2.txt', temp2, delimiter = '\n')  

plt.figure()
plt.xlim(0,1)
_ = plt.hist(temp3, bins=10000)
plt.savefig('savefile/hist/plain/' +'/histogram_encdec.png')
plt.clf() 
# np.savetxt('savefile/hist/plain/temp3.txt', temp3, delimiter = '\n')