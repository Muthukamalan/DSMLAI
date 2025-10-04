from typing import Callable,List,Any
from pathlib import Path

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS,EVAL_DATALOADERS
import torch
from torch.utils.data import DataLoader,random_split,default_collate
from torchvision import transforms
from torchvision.datasets import MNIST

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LitMNISTDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir:Path,
            batch_size:int = 32,
            num_workers:int = 0,
            test_transform:Callable  = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(.1307,),std=(.3081,))]),
            train_transform:Callable = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(.1307,),std=(.3081,))]),
            collate_fn:Callable      = default_collate
    ) -> None:
        super().__init__()
        self.data_dir:Path = Path('.') if data_dir is None else data_dir
        self.batch_size:int = batch_size
        self.num_workers:int = num_workers
        self.test_transform:Callable = test_transform
        self.train_transform:Callable = train_transform
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        MNIST(self.data_dir,train=True,download=True)
        MNIST(self.data_dir,train=False,download=True)
    
    def setup(self, stage: str=None) -> None:
        if stage=="fit" or stage is None:
            _mnist_full = MNIST(self.data_dir,train=True,transform=self.train_transform)
            self.mnist_train, self.mnist_val = random_split(_mnist_full,[.9,.1],generator=torch.Generator(device))
        
        if stage=='test' or stage is None: 
            self.mnist_test = MNIST(self.data_dir,train=False, transform=self.test_transform)

    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.mnist_train,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True,generator= torch.Generator(device) )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_val,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False,generator= torch.Generator(device))
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.mnist_test,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False,generator= torch.Generator(device))
    