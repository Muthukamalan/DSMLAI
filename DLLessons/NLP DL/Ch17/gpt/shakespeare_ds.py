from data_config import data_dir,data_url
from model_config import model_args


import os
import numpy as np 
import requests 
import tiktoken 
import torch

from typing import List

class ShakespeareDS(torch.utils.data.Dataset):
    def __init__(
            self,
            train:bool=False,
            data_dir=data_dir,
            data_url = data_url,
            **kwargs
    )->None:
        self.train = train
        self.block_size = model_args.get('init_args',1024).get('block_size')


        if not os.path.exists(os.path.join(data_dir,'input.txt')):
            ShakespeareDS.prepare(data_dir,data_url)

        self.data  = np.memmap(os.path.join(data_dir,'train.bin') if train else os.path.join(data_dir,'val.bin'))
        

    def __getitem__(self,idx)->List[torch.Tensor]:
        x = torch.from_numpy( (self.data[idx: idx+self.block_size]).astype(np.int64) )
        y = torch.from_numpy( (self.data[idx+1: idx+1+self.block_size]).astype(np.int64) )
        return x,y


    def __len__(self)->int:
        return len(self.data)
            


    

    @staticmethod
    def prepare(data_dir,data_url):
        input_file_path = os.path.join(data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            # data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()

            train_data = data[:int(.9*len(data))]
            val_data = data[int(.9*len(data)):]

        encoder = tiktoken.get_encoding('gpt2')
        train_ids = encoder.encode_ordinary(train_data)
        val_ids   = encoder.encode_ordinary(val_data)

        train_ids = np.array(train_ids,dtype=np.uint16)
        val_ids   = np.array(val_ids,dtype=np.uint16)

        train_ids.tofile(os.path.join(data_dir,'train.bin'))
        val_ids.tofile(os.path.join(data_dir,'val.bin'))
