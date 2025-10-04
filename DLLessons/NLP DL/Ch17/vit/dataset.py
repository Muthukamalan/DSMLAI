import os 
import torch
import torchvision
# import lightning as pl 
# import torch.utils.data
# from config import BATCH_SIZE,DEVICE,TEST_DIR,TRAIN_DIR,MANUAL_TRANSFORMS


def create_dataloaders(
        train_dir:str,
        test_dir:str,
        transformation:torchvision.transforms.Compose,
        batch_size:str,
        num_workers:int
)->tuple[torch.utils.data.DataLoader,torch.utils.data.DataLoader,list[str]]:
    
    train_data = torchvision.datasets.ImageFolder(train_dir,transform=transformation)
    test_data  = torchvision.datasets.ImageFolder(test_dir,transform=transformation)
    class_names:list[str] = train_data.classes


    train_dataloader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

    return train_dataloader,test_dataloader,class_names