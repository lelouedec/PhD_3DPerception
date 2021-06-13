import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy import spatial,ndimage

import scipy
import json
from matplotlib import cm as CM
# from image import *
import torch


L = 1
def locations_to_proba(density_map,gt):
    # sigma = 2
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    locations = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    locations2 = locations.copy()

    locations = (np.array(locations)/L).astype(np.uint64)
    ptsx1 = locations[:int(len(locations)/3),1]
    ptsy1 = locations[:int(len(locations)/3),0]

    ptsx2 = locations[int(len(locations)/3):int(len(locations)/3)*2,1]
    ptsy2 = locations[int(len(locations)/3):int(len(locations)/3)*2,0]

    ptsx3 = locations[int(len(locations)/3)*2:,1]
    ptsy3 = locations[int(len(locations)/3)*2:,0]



    tree = spatial.KDTree(locations2, leafsize=2048)
    distances, locat = tree.query(locations2, k=4)

    # print ('generate density...')
    sigmas = []
    for i, pt in enumerate(locations2):
        sigma = distances[i][1:].min()/L
        # if(sigma*0.02 < 2):
        sigmas.append(min(sigma*0.33,5))
        # else:
        #     sigmas.append(2.5)
    # exit()
    sigmas = np.array(sigmas)
    # sigmas = np.ones((len(locations)) ) * 2.5

    
    ii1,jj1 = np.ogrid[:density_map.shape[0],:density_map.shape[1]]  # memory-efficient np.indices


    peaks  = np.exp(-( (ptsx1[:,None,None] - ii1)**2) / (sigmas[:int(len(locations)/3),None,None]**2)) * np.exp(-( (ptsy1[:,None,None] - jj1)**2) / (  sigmas[:int(len(locations)/3),None,None]**2)) # shape (npeaks,*imsize)

    peaks2 = np.exp(-( (ptsx2[:,None,None] - ii1)**2) / (sigmas[int(len(locations)/3):int(len(locations)/3)*2,None,None]**2)) * np.exp(-( (ptsy2[:,None,None] - jj1)**2) / (  sigmas[int(len(locations)/3):int(len(locations)/3)*2,None,None]**2)) # shape (npeaks,*imsize)
    
    peaks3 = np.exp(-( (ptsx3[:,None,None] - ii1)**2) / (sigmas[int(len(locations)/3)*2:,None,None]**2)) * np.exp(-( (ptsy3[:,None,None] - jj1)**2) / (  sigmas[int(len(locations)/3)*2:,None,None]**2)) # shape (npeaks,*imsize)



    # sum each peak to end up with an array of size imsize
    peaks  = peaks.sum(axis=0)
    peaks2 = peaks2.sum(axis=0)
    peaks3 = peaks3.sum(axis=0)

    peaks = peaks + peaks2 + peaks3

    peaks = (peaks - peaks.min() )/ (peaks.max()-peaks.min())
    return peaks




root = "./shangai"
#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train,part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)



for img_path in img_paths:
    print (img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = locations_to_proba(np.zeros(img.shape),k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

path_sets = [part_B_train,part_B_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print (img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = locations_to_proba(np.zeros(img.shape),k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k