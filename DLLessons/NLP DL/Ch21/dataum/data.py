import os
import torch
import lightning as pl 
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device=device)

class DSTorch(torch.utils.data.Dataset):
    def __init__(self,train:bool,block_size:int,file_path:str)->None:
        self.train=train
        self.block_size = block_size
        
        assert os.path.isfile(file_path),f"File not exists in file_path:{file_path}"
        self._prepare(file_path=file_path)
    
    def _prepare(self,file_path):
        with open(file_path,'r',encoding='utf-8') as f: self.data=f.read()

        self.train_data = self.data[:int(.9*len(self.data))]
        self.test_data = self.data[int(.9*len(self.data)):]

        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)

        ctoi = {c:id for id,c in enumerate(self.chars)}
        itoc  = {id:c for id,c in enumerate(self.chars)}

        self.encode = lambda x: [ctoi[ch] for ch in x]                #input string, output list of ints
        self.decode = lambda ids: "".join([itoc[id] for id in ids])

        self.train_ds:torch.Tensor = torch.tensor(self.encode(self.train_data))
        self.test_ds:torch.Tensor  = torch.tensor(self.encode(self.test_data))

       

    def __len__(self)->int:
        if self.train==True:
            return len(self.train_ds)-self.block_size-1
        else:
            return len(self.test_ds)-self.block_size-1

    def __getitem__(self,idx):
        if self.train==True:
            return self.train_ds[idx:idx+self.block_size], self.train_ds[idx+1:idx+1+self.block_size]
        else:
            return self.test_ds[idx:idx+self.block_size], self.test_ds[idx+1:idx+1+self.block_size]
        



class LitAuthorData(pl.LightningDataModule):
    def __init__(
            self,
            file_path:str,
            block_size:int,
            batch_size:int,
            num_workers:int
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.block_size:int = block_size
        self.file_path:int  = file_path
        self.batch_size:int = batch_size
        self.num_workers:int= num_workers

    def prepare_data(self) -> None:
        ...
    
    def setup(self, stage: str=None) -> None:
        self.train_ds:DSTorch = DSTorch(train=True,block_size=self.block_size,file_path=self.file_path)
        self.test_ds:DSTorch  = DSTorch(train=False,block_size=self.block_size,file_path=self.file_path)
        self.data_decode      = self.train_ds.decode
        self.data_encode      = self.train_ds.encode
        self.data             = self.train_ds.data
        
    
    def decoder(self,seq:list|torch.Tensor):
        '''decoder property to handle 1D list or tensor DType'''
        if isinstance(seq,list):
            return self.data_decode(seq)
        elif isinstance(seq,torch.Tensor):
            return self.data_decode(seq.tolist())
        
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=torch.Generator('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=torch.Generator('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    
## GPT2 Encoder
import tiktoken
class GPT2DSTorch(torch.utils.data.Dataset):
    def __init__(self,train:bool,block_size:int,file_path:str,encoder='gpt2'):
        self.train = train
        self.block_size= block_size
        self.enc  = tiktoken.get_encoding(encoder)
        self._prepare(file_path=file_path)
    
    def _prepare(self,file_path):
        with open(file_path,'r',encoding='utf-8') as f: self.data=f.read()

        self.train_data = self.data[:int(.9*len(self.data))]
        self.test_data = self.data[int(.9*len(self.data)):]

        self.vocab_size = self.enc.n_vocab

        self.train_ds = torch.tensor(self.enc.encode(self.train_data, allowed_special={"<|endoftext|>"}))
        self.test_ds = torch.tensor(self.enc.encode(self.test_data, allowed_special={"<|endoftext|>"}))

    def decode(self,seq):
        if isinstance(seq,torch.Tensor):
            return self.enc.decode(seq.tolist())
        elif isinstance(seq,list):
            return self.enc.decode(seq)

    def  __len__(self):
        if self.train:
            return len(self.train_ds)-self.block_size-1
        else:
            return len(self.test_ds)-self.block_size-1
        
    def __getitem__(self,idx):
        if self.train:
            return (self.train_ds[idx:idx+self.block_size], self.train_ds[idx+1:idx+1+self.block_size] )
        else:
            return (self.test_ds[idx:idx+self.block_size], self.test_ds[idx+1:idx+1+self.block_size] )
    

class GPT2LitAuthorData(pl.LightningDataModule):
    def __init__(
            self,
            file_path:str,
            block_size:int,
            batch_size:int,
            num_workers:int,
            encoder:str = 'gpt2'
    ) -> None:
        super().__init__()
        self.block_size:int = block_size
        self.file_path:int  = file_path
        self.batch_size:int = batch_size
        self.num_workers:int= num_workers
        self.encoder:str    = encoder
        

    def prepare_data(self) -> None:
        ...
    
    def setup(self, stage: str=None) -> None:
        self.train_ds:GPT2DSTorch = GPT2DSTorch(train=True,block_size=self.block_size,file_path=self.file_path,encoder=self.encoder)
        self.test_ds:GPT2DSTorch  = GPT2DSTorch(train=False,block_size=self.block_size,file_path=self.file_path,encoder=self.encoder)
        
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=torch.Generator('cuda' if torch.cuda.is_available() else 'cpu')
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=torch.Generator('cuda' if torch.cuda.is_available() else 'cpu')
        )