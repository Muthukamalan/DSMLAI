import os
import cv2
import pandas as pd
import torch
import torchvision
import albumentations as A
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split
import config as CFG



def get_transforms()->A.Compose:
    return A.Compose([
            A.Resize(height=CFG.size,width=CFG.size,always_apply=True),
            A.Normalize(max_pixel_value=255.0,always_apply=True)
        ])

class CLIPDataset(Dataset):
    def __init__(self,
                 image_filenames:str, 
                 captions:list[str], 
                 tokenizer, 
                 transforms:torchvision.transforms.Compose
    ) -> None:
        super().__init__()
        self.image_filenames:pd.Series = image_filenames
        self.captions:pd.Series        = captions
        self.transforms      = transforms

        self.encoded_captions:dict = tokenizer(
                                    [str(i) for i in self.captions.tolist()], # self.captions,
                                    padding=True,
                                    truncation=True,
                                    max_length=CFG.max_length
        )


    def __getitem__(self, index) ->dict:
        item = { k:torch.tensor(v[index]) for k,v in self.encoded_captions.items() }

        img = torchvision.io.read_image(os.path.join( f"{CFG.image_path}", self.image_filenames[index] ))
        img = self.transforms(image=img.permute(1,2,0).numpy())['image']
        
        item['image'] = torch.tensor(img).permute(2,0,1).float()   # returns <<< ( channels, height, width )
        item['caption'] = self.captions[index]

        return item
    
    def __len__(self): return len(self.captions)

