from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import os 
import torchvision
import lightning as pl 
from PIL import Image


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
            path=os.path.join(r"C:\Users\muthu\GitHub\DATA 📁"), 
            split="trainval", 
            target_types="segmentation", 
        transforms:torchvision.transforms.Compose=None, 
        mask_transforms:torchvision.transforms.Compose=None
    ) -> None:
        super(OxfordPetDataset,self).__init__()
        self.data = torchvision.datasets.OxfordIIITPet(root=path, split=split, target_types=target_types)
        self.transforms:torchvision.transforms.Compose = transforms
        self.mask_transforms:torchvision.transforms.Compose = mask_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor,torch.Tensor]:
        img  = Image.open(self.data._images[idx]).convert("RGB")
        lbl  = Image.open(self.data._segs[idx]) 

        img  = self.transforms(img)
        lbl  = self.mask_transforms(lbl)
        one_hot_label = torch.nn.functional.one_hot(lbl,3).transpose(0,2)
        return img,lbl,one_hot_label


class OxfordPetDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size:int=16 ,
            path=os.path.join(r"C:\Users\muthu\GitHub\DATA 📁"), 
            transforms:None|torchvision.transforms.Compose = None ,
            mask_transforms:None|torchvision.transforms.Compose = None, 
    ) -> None:
        super().__init__()
        
        self.path:str = path 
        
        assert os.path.isdir(self.path),"folder not found"
        
        self.transform:None|torchvision.transforms.Compose = transforms
        self.mask_transform:None|torchvision.transforms.Compose = mask_transforms

        self.batch_size:int = batch_size 


    def setup(self, stage: str=None) -> None:
        self.train_dataset = OxfordPetDataset(
            path= self.path,
            split='trainval',
            transforms=self.transform,
            mask_transforms=self.mask_transform,
        )
        self.val_dataset = OxfordPetDataset(
            path= self.path,
            split='test',
            transforms=self.transform,
            mask_transforms=self.mask_transform
        )
    
    def train_dataloader(self) -> torch.Any:
        return torch.utils.data.DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,shuffle=True)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(dataset=self.val_dataset,batch_size=self.batch_size,shuffle=False)
    