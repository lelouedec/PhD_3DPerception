import open3d
import torch
import torch.nn as nn 
import tqdm
import dataset_class
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import metrics
import time
from PIL import Image

from sklearn import metrics as m

def auc(modelpath,idxlist,datasetpath,for_all=False):
    model = torch.load(modelpath).cuda().eval()
   

    dataset = dataset_class.Loading_dataset(idxlist,datasetpath,False,False)
    my_dataset_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=1)

    eth = 0.5
    print("testing..")
    precisions = [0.0]
    recalls = [1.0]
    while(eth<0.95):
        idfix = 0
        ious = 0
        kappas = 0
        accs   = 0
        tps   = 0
        fps   = 0
        precs = 0
        recs  = 0
        for sample in my_dataset_loader:
            pts_tensor     = sample['pts'].float().permute(0,3,1,2)
            normals_tensor = sample['normals'].float().permute(0,3,1,2)
            colors_tensor  = sample['colors'].float().permute(0,3,1,2)
            target_tensor      = sample['gt']
            input_tensor = torch.cat([pts_tensor,normals_tensor],1).cuda()
            # input_tensor = torch.cat([pts_tensor,normals_tensor,colors_tensor],1).cuda()
            # input_tensor  = colors_tensor.cuda()
            # input_tensor = pts_tensor.cuda()
            target_tensor = target_tensor.to(device="cuda", dtype=torch.long)
            
            with torch.no_grad():
                out  = model(input_tensor)
            softmax_mx = out.detach().permute(0,2,3,1).cpu().numpy()[0]

            softmax_mx[softmax_mx[:,:,1]<eth]=0.0
            softmax_mx[softmax_mx[:,:,1]!=0.0]=1.0

            pts = pts_tensor.detach().permute(0,2,3,1).numpy()[0]
            pts2 = pts.reshape((pts.shape[0]*pts.shape[1],3))
            idxs   = np.any(pts2 != [0.0, 0.0, 0.0], axis=-1)
            
            pred  =  softmax_mx[:,:,1]
            gta   =  target_tensor.cpu().numpy()[0]
            pred  = pred.reshape((pred.shape[0]*pred.shape[1]))
            gta   = gta.reshape((gta.shape[0]*gta.shape[1]))
            pred  = pred[idxs]
            gta   = gta[idxs]

            union =  np.logical_or(pred,gta)
            intersection = np.logical_and(pred,gta)
            iou = intersection.sum()/union.sum()       
            
            ious = ious + iou
            kappa =  metrics.Kappa_cohen(pred,gta)
            acc   =  metrics.Accuracy(pred,gta)
            something  = metrics.precision_recall(pred,gta)
            tps = something[2]/(something[2]+something[5]) + tps
            fps = something[3]/(something[4]+something[3]) + fps
            precs = precs + something[0]
            recs = recs + something[1]
            kappas = kappas + kappa
            accs = accs + acc
            idfix = idfix + 1
        
        precisions.append(precs/idfix)
        recalls.append(recs/idfix)
        print(accs/idfix,kappas/idfix,ious/idfix)
        eth = eth+0.05
    precisions.append(1.0)
    recalls.append(0.0)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    print(precisions,recalls)
    print(m.auc(recalls, precisions))
    print("TPS",tps/idfix)
    print("FPS",fps/idfix)
    print("MIoU",ious/idfix)
    print("kappas",kappas/idfix)
    print("accuracy",accs/idfix)
    print("precision",precs/idfix)
    print("recall",recs/idfix)
    
            




if __name__ == "__main__":
    
    auc("./models/cnn_simu_reflectance2.ckpt",["test_simu.txt"],["Data"],for_all=False)
    
    

