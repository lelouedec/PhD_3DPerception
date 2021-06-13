import model as M
import modelbis as M2
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt 
import cv2
from torch.utils.data import Dataset, DataLoader
import glob 
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import random
import dataset as dt
import dilated_people

import os
import torchvision.transforms.functional as F
from image import *
class listDataset(Dataset):
    def __init__(self, root,training):
        if(training):
            self.lines = root+"train_data/"
        else:
            self.lines = root+"test_data/"
        self.transforms = transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        img,target = load_data(img_path,self.train)
                
        
        if self.transform is not None:
            img = self.transform(img)
        return img,target



model = dilated_people.Terminator(3,1)
lr = 1e-6

criterion = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
dataset = listDataset("shangai/part_A_final/",True)
train_loader = torch.utils.data.DataLoader(dataset,batch_size=4,num_workers=2)

dataset2 = listDataset("shangai/part_A_final/",True)
train_loader2 = torch.utils.data.DataLoader(dataset2,batch_size=4,num_workers=2)

writer = SummaryWriter()
for epoch in range(300):
    lost = []
    for v in dataloader:
        input_tensor = v[0].cuda()
        target = v[1].cuda()

        output = model(input_tensor.float(),(input_tensor.shape[2],input_tensor.shape[3]))

        loss = criterion(output[0].double().squeeze(1), target.double())

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lost.append(loss.detach().cpu().item())
    lost2 = []
    if(epoch%2==0 and epoch!=0):
        for v in dataloader2:
            input_tensor = v[0].cuda()
            target = v[1].cuda()

            output = model(input_tensor.float(),(input_tensor.shape[2],input_tensor.shape[3]))
            loss = criterion(output[0].double().squeeze(1), target.double())

            lost2.append(loss.detach().cpu().item())
        writer.add_scalar("Loss testing",np.array(lost2).mean(0),epoch)
    if(epoch%2-==0 and epoch!=0):
        torch.save(model, "dilated_sh_temp.ckpt")


    writer.add_scalar("Loss",np.array(lost).mean(0),epoch)
torch.save(model, "dilated_sh.ckpt")


