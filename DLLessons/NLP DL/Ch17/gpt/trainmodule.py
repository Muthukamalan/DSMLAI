from typing import Optional,Any

import torch 
import pytorch_lightning as pl 

from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Dict
from model import NanoGPT
from model_config import model_args


class LitNanoGPT(pl.LightningModule):
    def __init__(self,inital_args:Dict, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net:NanoGPT = NanoGPT(config=model_args)
        self.loss_fn     = torch.nn.functional.cross_entropy

    def forward(self, batch, *args: Any, **kwargs: Any) -> Any:
        x,y = batch
        logits = self.net(x)
        loss  = self.loss_fn(
                        logits.view(-1,logits.size(-1)),
                        y.view(-1),
                        ignore_index=-1
                )
        return loss 
    
    def training_step(self,batch, *args: Any, **kwargs: Any) -> torch.Tensor :
        loss = self(batch)
        self.log(name='train_loss',value=loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss 
    
    def test_step(self,batch, *args: Any, **kwargs: Any) -> torch.Tensor :
        loss = self(batch)
        self.log(name='test_loss',value=loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss 
    
    def validation_step(self,batch, *args: Any, **kwargs: Any) -> torch.Tensor :
        loss = self(batch)
        self.log(name='val_loss',value=loss,prog_bar=True,on_step=True,on_epoch=True)
        return loss 
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters())
    