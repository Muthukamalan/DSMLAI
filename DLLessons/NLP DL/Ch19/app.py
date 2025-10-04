from lightning.pytorch.utilities.types import TRAIN_DATALOADERS,EVAL_DATALOADERS
import pandas as pd
import lightning as pl 
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data
from transformers import DistilBertTokenizerFast


from modules import ImageEncoder,TextEncoder,ProjectionHead
from dataset import CLIPDataset,get_transforms
import config as CFG

class Flickerlit(pl.LightningDataModule):
    def __init__( 
            self, 
            caption_path:str  = CFG.captions_path,
            text_tokenizer:str= CFG.text_tokenizer,
            transforms        = get_transforms,
            batch_size        = CFG.batch_size,
            num_workers       = CFG.num_workers
    ) -> None:
        super().__init__()
        self.train_data:pd.DataFrame
        self.val_data:  pd.DataFrame
        self.train_data,self.val_data = train_test_split(pd.read_csv(caption_path),test_size=0.3,shuffle=True,random_state=123)
        self.train_data.reset_index(drop=True,inplace=True)
        self.val_data.reset_index(drop=True,inplace=True)
        self.tokenizer:DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path= text_tokenizer,
        )
        self.transform = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass 

    def setup(self, stage: str='trainval') -> None:
        self.train_ds:CLIPDataset = CLIPDataset(
                                    image_filenames=self.train_data['image'],
                                    captions=self.train_data['caption'],
                                    tokenizer= self.tokenizer,
                                    transforms= self.transform()
                                ) 
        
        self.val_ds:CLIPDataset = CLIPDataset(
                                    image_filenames=self.val_data['image'],
                                    captions=self.val_data['caption'],
                                    tokenizer= self.tokenizer,
                                    transforms= self.transform()
                                ) 
        
    def train_dataloader(self)->TRAIN_DATALOADERS:
        return torch.utils.data.DataLoader(self.train_ds,batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.val_ds,batch_size=self.batch_size, shuffle=False)
    


class LitFlick(pl.LightningModule):
    def __init__(
            self, 
            temperature:int  = CFG.temperature,
            image_embedding:str = CFG.image_embedding,
            text_embedidng:str  = CFG.text_embedding,
            *args: TRAIN_DATALOADERS, 
            **kwargs: TRAIN_DATALOADERS
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_encoder:ImageEncoder    = ImageEncoder()
        self.text_encoder:TextEncoder      = TextEncoder()
        self.img_projection:ProjectionHead = ProjectionHead(embedding_dim=image_embedding)
        self.txt_projection:ProjectionHead = ProjectionHead(embedding_dim=text_embedidng)
        self.temperature:int               = temperature


    def forward(self,batch:dict, *args: TRAIN_DATALOADERS, **kwargs: TRAIN_DATALOADERS) -> TRAIN_DATALOADERS:
        img_features = self.image_encoder(batch['image'])
        txt_features = self.text_encoder(batch['input_ids'],batch['attention_mask'])

        img_embedding = self.img_projection(img_features)
        txt_embedding = self.txt_projection(txt_features)

        logits = torch.matmul(img_embedding,txt_embedding.T) / self.temperature

        img_similarity = img_embedding@img_embedding.T
        txt_similarity = txt_embedding@txt_embedding.T

        targets = torch.nn.functional.softmax( (img_similarity+txt_similarity)/ 2*self.temperature, dim=-1 )

        txt_loss = self.cross_entropy(logits,targets,reduction='none')
        img_loss = self.cross_entropy(logits.T,targets.T,reduction='none')
        
        loss     = (txt_loss+img_loss)/2.
        return loss.mean()

    

    def cross_entropy(self,preds:torch.Tensor,targets:torch.Tensor,reduction='none'):
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        loss        = (-targets * log_softmax(preds)).sum(1)
        if reduction=='none':
            return loss 
        return loss.mean()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')
        return {
            'optimizer':optimizer,
            'lr_scheduler':{
                'scheduler':lr_scheduler,
                'interval':'step',
                'frequency':1,
                'monitor':'val_loss',
                'strict':False
            }
        }
        

    def training_step(self,batch,batch_idx, *args: TRAIN_DATALOADERS, **kwargs: TRAIN_DATALOADERS) -> torch.Tensor:
        loss = self(batch)
        self.log('train_loss',loss,prog_bar=True,logger=True,on_epoch=True,on_step=True)
        return loss 
    

    def validation_step(self,batch,batch_idx, *args: TRAIN_DATALOADERS, **kwargs: TRAIN_DATALOADERS) -> torch.Tensor:
        loss = self(batch)
        self.log('val_loss',loss,prog_bar=True,logger=True,on_epoch=True,on_step=True)
        return loss 
    