import torch
import torch.nn as nn 
import tqdm
import dataset_class
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import CNN3D

def train_function():
    model = CNN3D.CNN3D(input_channels=6, output_channels=2).train().cuda()

    ######HYPER PARAMETERS
    lr = 1e-5
    epochs = 500 #to adapt depending on training choosen
    
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr }])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,16]).float()).cuda()
    
    for_all=False  #set to true if you want to train on all data found in the folder and ignore indices
    dataset = dataset_class.Loading_dataset("path_to_ids_list_file","path_to_dataset_folder",for_all,True)
    my_dataset_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)

    
    print("training..")
    writer = SummaryWriter("runs/model_name")
    for p in tqdm.tqdm(range(0,epochs)):
        loss_f = []
        for sample in my_dataset_loader:
            pts_tensor     = sample['pts'].float().permute(0,3,1,2)
            normals_tensor = sample['normals'].float().permute(0,3,1,2)
            colors_tensor  = sample['colors'].float().permute(0,3,1,2)
            target_tensor      = sample['gt']

            input_tensor = torch.cat([pts_tensor,normals_tensor],1).to(device="cuda", dtype=torch.float32)
            # input_tensor  = colors_tensor.cuda()
            # input_tensor  = pts_tensor.cuda()

            target_tensor = target_tensor.to(device="cuda", dtype=torch.long)
            out = model(input_tensor)
            loss = criterion(out,target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_f.append(loss.float().cpu().detach().numpy().mean())
            del loss
            del input_tensor
            del out
            del target_tensor
        writer.add_scalar('Loss',np.array(loss_f).mean(), p)
        
    torch.save(model, "models/model.ckpt")


if __name__ == "__main__":
    train_function()
