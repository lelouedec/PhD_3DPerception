import json 
import csv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial,ndimage

L=5

# Learning To Count Objects in Images
def locations_to_proba(density_map,locations):
    # sigma = 2
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
        sigmas.append(min(sigma*0.33,2.5))
    sigmas = np.array(sigmas)
    # sigmas = np.ones((len(locations)) ) * 2.5

    
    ii1,jj1 = np.ogrid[:density_map.shape[0],:density_map.shape[1]]  # memory-efficient np.indices


    peaks  = np.exp(-( (ptsx1[:,None,None] - ii1)**2) / (sigmas[:int(len(locations)/3),None,None]**2)) * np.exp(-( (ptsy1[:,None,None] - jj1)**2) / (  sigmas[:int(len(locations)/3),None,None]**2)) # shape (npeaks,*imsize)

    peaks2 = np.exp(-( (ptsx2[:,None,None] - ii1)**2) / (sigmas[int(len(locations)/3):int(len(locations)/3)*2,None,None]**2)) * np.exp(-( (ptsy2[:,None,None] - jj1)**2) / (  sigmas[int(len(locations)/3):int(len(locations)/3)*2,None,None]**2)) # shape (npeaks,*imsize)
    
    peaks3 = np.exp(-( (ptsx3[:,None,None] - ii1)**2) / (sigmas[int(len(locations)/3)*2:,None,None]**2)) * np.exp(-( (ptsy3[:,None,None] - jj1)**2) / (  sigmas[int(len(locations)/3)*2:,None,None]**2)) # shape (npeaks,*imsize)



    # sum each peak to end up with an array of size imsize
    peaks = peaks.sum(axis=0)
    peaks2 = peaks2.sum(axis=0)
    peaks3 = peaks3.sum(axis=0)

    peaks = peaks + peaks2 + peaks3

    peaks = (peaks - peaks.min() )/ (peaks.max()-peaks.min())
    return peaks


def locations_density2(locations):
    density = np.zeros((int(950/L),int(3600/L)))
    locations = (np.array(locations)/L).astype(np.uint16)
    ptsx = locations[:,1]
    ptsy = locations[:,0]
    pts = np.array(list(zip(ptsx, ptsy)))
    print(pts.shape)
    tree = spatial.KDTree(pts.copy(), leafsize=2048)
    # query kdtree
    gt_count = len(locations)
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(density.shape, dtype=np.float32)
        pt2d[max(0,min(pt[0],int(950/L)-1)),max(0,min(pt[1],int(3600/L)-1))] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(density.shape))/2./2. #case: 1 point
        density += ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    print(density.sum())
    return density


lines = []
with open("via_region_data.csv") as f :
    csv_reader = csv.reader(f, delimiter=',')
    for row in csv_reader:
        lines.append(row)

lists_annot = []
img_index = "0400.jpg"
img_annots = []
imgs = ["0400"]
for i in range(1,len(lines)):
    new_img_index = lines[i][0]
    if(new_img_index==img_index and i!=len(lines)-1):
        pts = json.loads(lines[i][5])
        cx = pts["cx"]
        cy = pts["cy"]
        img_annots.append([cx,cy])
    elif(i==len(lines)-1):
        pts = json.loads(lines[i][5])
        cx = pts["cx"]
        cy = pts["cy"]
        img_annots.append([cx,cy])
        lists_annot.append(img_annots)
        imgs.append(new_img_index[:-4])
    else:
        img_index=new_img_index
        lists_annot.append(img_annots)
        imgs.append(new_img_index[:-4])
        img_annots = []
        pts = json.loads(lines[i][5])
        cx = pts["cx"]
        cy = pts["cy"]
        img_annots.append([cx,cy])





print(imgs,len(lists_annot))
nb_seeds = []
for i,l in enumerate(lists_annot):
    l = np.unique(l,axis=0)
    mask         = locations_to_proba(np.zeros((int(950/L),int(3600/L))),l)
    mask_density = locations_density2(l)
    np.save("./densities/"+imgs[i]+".npy",mask_density)

    mask[mask<0.01] = 0.0
    image = Image.fromarray(mask*255).convert("L")
    image.save("./annotations/"+imgs[i]+".jpg")
    nb_seeds.append(len(l))

np.savetxt("nb_seeds.csv", np.array(nb_seeds).astype(np.uint16),delimiter="'")


