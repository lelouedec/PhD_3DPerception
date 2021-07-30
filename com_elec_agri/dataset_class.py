import open3d
import data as data_utils
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from scipy import ndimage, misc
import PIL.Image as Image
import glob 

### This function will load all paths with their annotation and names and return it 
def load_simu_data_paths(idx_file,dataset_path,all=False,train=True,nb_samples=134):
    if(all):
        idx = np.arange(0,nb_samples)
    else:
        idx = list(np.genfromtxt(idx_file).astype(int))
    
    print(dataset_path+"/pointclouds/*")
    pc        = np.array(sorted(glob.glob(dataset_path+"/pointclouds/*")))[idx]
    segmentations = np.array(sorted(glob.glob(dataset_path+"/binary/*")))[idx]
    data = []
    annotations = []
    names = []
    for i,c in enumerate(pc):
        annotations.append(segmentations[i])
        data.append(c)
        names.append(c)
    
    return data,annotations,names

class Loading_dataset(Dataset):
    def __init__(self,idxs,datasetpath,for_all,training):
        self.data = []
        self.annotations = []
        self.names = []
        self.training = training
        self.transform = None
        for indices in range(0,len(idxs)):
            self.data1,self.annotations1,self.names1 = load_simu_data_paths(idxs[indices],datasetpath[indices],for_all,training)
            self.data            = self.data            + self.data1
            self.annotations     = self.annotations     + self.annotations1
            self.names           = self.names           + self.names1
        print("We have ",len(self.data)," samples")
        self.display = False
        self.width = 530
        self.height = 871

    def __getitem__(self, index): #load data into memory, pytorch loader will do the threading and batching
        pc = open3d.io.read_point_cloud(self.data[index])
        points  = np.asarray(pc.points)#*[1,1,-1]
        colors  = np.asarray(pc.colors)
        normals = np.asarray(pc.normals)
        colors  = np.reshape(colors,(self.width, self.height,3))
        points  = np.reshape(points,(self.width, self.height,3))
        normals = np.reshape(normals,(self.width, self.height,3))
        annotation = np.reshape(np.load(self.annotations[index]),(self.width, self.height))


        padded_points    = np.zeros((points.shape[1],points.shape[1],3))
        padded_colors    = np.zeros((points.shape[1],points.shape[1],3))
        padded_normals   = np.zeros((points.shape[1],points.shape[1],3))
        padded_GT        = np.zeros((points.shape[1],points.shape[1]))

        padded_points [:points.shape[0],:points.shape[1],:] = points
        padded_colors [:points.shape[0],:points.shape[1],:] = colors
        padded_normals[:points.shape[0],:points.shape[1],:] = normals
        padded_GT     [:points.shape[0],:points.shape[1]] = annotation


        if(self.training):

            angle   = random.randrange(-180,180)
            min_p   = points.min(0).min(0)
            max_p   = points.max(0).max(0)
            x_shift = random.uniform(min_p[0],max_p[0])
            y_shift = random.uniform(min_p[1],max_p[1])
            z_shift = random.uniform(min_p[2],max_p[2])
            angle2 = angle*np.pi/180
            rotation_matrix = np.array([
                                        [np.cos(angle2),-np.sin(angle2),0],
                                        [np.sin(angle2),np.cos(angle2),0],
                                        [0,0,1]])
            padded_points  = np.dot(padded_points,rotation_matrix)
            padded_points  = ndimage.rotate(padded_points,  angle  ,reshape=False ,order=0) + [x_shift,y_shift,z_shift]


            padded_normals  = np.dot(padded_normals,rotation_matrix)
            padded_normals = ndimage.rotate(padded_normals, angle  ,reshape=False ,order=0)
            padded_colors  = ndimage.rotate(padded_colors,  angle  ,reshape=False ,order=0)
            padded_GT      = ndimage.rotate(padded_GT,      angle  ,reshape=False ,order=0)
        

        sample = {'pts': torch.from_numpy(padded_points),
                  'normals': torch.from_numpy(padded_normals),
                  'colors': torch.from_numpy(padded_colors),
                  'gt': torch.from_numpy(padded_GT),
                  'names':self.names[index]}

        return sample


    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    dataset = Loading_dataset(["all"],["./Data_perfect"],for_all=True,training=True)
    data = dataset.__getitem__(2)
    # print(data)
