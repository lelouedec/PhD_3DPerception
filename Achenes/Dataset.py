from torch.utils.data import Dataset
import glob 
import numpy as np 
import random
import torch
from PIL import Image



class Projections(Dataset):

    def __init__(self,idxs,training,width=360,height=95):
        self.paths  = sorted(glob.glob("./annotations/*.jpg"))
        self.paths  = list(np.array(self.paths)[idxs])
        self.training = training
        self.width  = width
        self.height = height
       
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
       

        input_tensor  = np.asarray(Image.open("./projections_x/"+self.paths[idx][len("./annotations/"):]).resize((self.width,self.height)))/255
        input_tensor2 = np.asarray(Image.open("./projections_y/"+self.paths[idx][len("./annotations/"):]).resize((self.width,self.height)))/255
        input_tensor3 = np.asarray(Image.open("./projections_z/"+self.paths[idx][len("./annotations/"):]).resize((self.width,self.height)))/255
        input_tensor  = np.concatenate([np.expand_dims(input_tensor,0),np.expand_dims(input_tensor2,0),np.expand_dims(input_tensor3,0)],0)


        target        =  np.asarray(Image.open(self.paths[idx]))/255 ##annotation was transformed to gray images

        if(self.training):
            translate    = random.randint(0, self.width)
            input_tensor = np.roll(input_tensor,translate,2)
            target = np.roll(target,translate,1)
            density = np.roll(density,translate,1)
        return torch.from_numpy(input_tensor),torch.from_numpy(target)


if __name__ == "__main__":
    dataset = Projections([0,1,2,3],False)
    for i in range(0,3):
        dataset.__getitem__(i)
