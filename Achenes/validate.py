import model as m
import numpy as np 
import torch
from torch import nn
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from skimage import measure
import matplotlib
from sklearn.metrics import mean_absolute_error,mean_squared_error
import dataset as dt
import time
import model as ourmodel

dataset2 = dt.Projections([  0,   1,   2,  14,  17,  20,  31,  36,  44,  54,  57,  59,  61,
        67,  69,  73,  76,  79,  92,  94, 100, 108, 112, 115, 123, 135,
       141, 142, 152, 155, 156, 157, 159, 160, 162, 175, 187, 195, 206,
       208, 213, 218, 226, 228, 238, 240, 241, 242, 254, 255, 264, 269,
       271, 274, 276, 278, 281, 286, 290, 293, 296, 299, 300, 302, 304,
       315, 316, 320, 321, 325, 330, 346, 356, 360, 364, 383, 386, 388,
       391, 395, 397, 405, 406, 414, 415, 417, 419, 427, 431, 435, 437,
       439, 448, 453, 454, 461, 470, 476, 482, 492, 495, 497, 501, 511,
       518, 519, 522, 524, 527, 528, 538, 546, 548, 549, 557, 558, 564,
       567, 570, 582, 583, 586, 591, 597, 602, 607, 612, 614, 615, 621,
       631, 637, 641, 648, 661, 667, 670, 671, 675, 682, 685, 697, 699,
       701, 706, 707, 713, 718, 737, 743, 747, 749, 755, 757, 767, 770],False,width=720,height=190)

model  = ourmodel.Terminator(3,1).cuda()
model.load_state_dict(torch.load("unet_trans.ckpt"))

dataloader  = DataLoader(dataset2, batch_size=1,shuffle=False, num_workers=1)

thresholds = [0.33]

ots = []
gts = []
for t in thresholds:
    gt = []
    ot = []
    for i,v in enumerate(dataloader):
        input_tensor = v[0].cuda()
        target  = v[1]

        seeds = v[2]

        output = model(input_tensor.float(),None)
        output = output.detach()[0][0].cpu().numpy()

        input_tensor = input_tensor.detach()[0].cpu().numpy().swapaxes(0,2).swapaxes(0,1)



        output2 = np.expand_dims(output,2)
        input_tensor2 = np.expand_dims(input_tensor,2)

        output2       = np.concatenate([output2,np.zeros(output2.shape),np.zeros(output2.shape)],2)
        input_tensor2 = np.concatenate([input_tensor2,input_tensor2,input_tensor2],2)

        outputbis = output.copy()
        outputbis[output>t] = 1.0
        outputbis[outputbis!=1.0] = 0.0

        
        
        blobs_labels,cnt_output = measure.label(outputbis, background=0,return_num=True)


        cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3))
        cmap.colors[0] = np.zeros((3))

        ot.append(cnt_output)
        gt.append(seeds)
       
        cmap2 = plt.cm.jet
        norm = plt.Normalize(vmin=output.min(), vmax=output.max())
        image = cmap2(norm(output))

        fig,ax = plt.subplots(nrows=3,ncols=1)
        ax[0].imshow(input_tensor)
        ax[0].set_title(str(int(seeds)) +":"+v[-1][0]+"gt"+str(gt[-1])+ ",pred "+str(ot[-1]))
        ax[1].imshow(image)
        ax[2].imshow(np.concatenate([input_tensor[:,:,2][:,:,None],input_tensor[:,:,2][:,:,None],input_tensor[:,:,2][:,:,None]],2)+\
                     np.concatenate([output[:,:,None],np.zeros(output[:,:,None].shape),output[:,:,None]],2))
        plt.show()

    ot = np.array(ot)
    gt = np.array(gt)

    ots.append(ot)
    gts.append(gt)



MSEs = []
MAEs = []
ACCs = []
FNs  = []
Fps  = []
Tps  = []
for i,t in enumerate(thresholds):
    ACCs.append( 1-(np.abs(gts[i]-ots[i])/gts[i]).mean())
    MAEs.append(mean_absolute_error(gts[i], ots[i]))
    MSEs.append(mean_squared_error(gts[i],ots[i],squared=False))
    res = gts[i]-ots[i]
    FNS = (gts[i]-ots[i])>0
    FPS = (gts[i]-ots[i])<0
    FNs.append( res[FNS].sum() )
    Fps.append( np.abs(res[FPS].sum()) )
    Tps.append(gts[i].sum()-np.abs(res[FNS].sum()))
print("MSE : ",MSEs)
print("MAE : ",MAEs)
print("FN : ",FNs)
print("FP : ",Fps)
print("TP : ",Tps)
print("-------")
