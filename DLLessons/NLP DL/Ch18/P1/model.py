import torch    
import lightning.pytorch as pl
from utils import DiceLoss

class Encoder(torch.nn.Module): 
    def __init__(self,inchannels:int ,outchannels:int, contraction_mode:str,is_downsample:bool=True,dropout_rate:float=0.01, *args, **kwargs) -> None:
        super(Encoder,self).__init__(*args, **kwargs)

        assert contraction_mode in ['maxpool','strided_conv'], "Maxpool and Strided Conv values are allowed to downsample"
        self.contraction_mode:str = contraction_mode
        # usually we do downsample, not when couple with decoder
        self.is_downsample:bool   = is_downsample
        self.conv = torch.nn.Sequential(
            # ( bs, C, H, W ) >> (bs, C, H, W )
            torch.nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(outchannels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(dropout_rate),
            torch.nn.Conv2d(outchannels,outchannels,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(outchannels),
            torch.nn.ReLU(inplace=True)
        )
        if self.contraction_mode=="maxpool":
            # aka MAXPOOL  (bs,C,H,W) >> ( bs, C, H//2 , W//2 )
            self.downsample = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        elif self.contraction_mode=='strided_conv':
            self.downsample = torch.nn.Conv2d( outchannels,outchannels,kernel_size=2,stride=2 )


    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.conv(x)
        skip = x 
        if self.is_downsample==True:
            x = self.downsample(x)
        return (x,skip)



class Decoder(torch.nn.Module):
    def __init__(self,inchannels:int, outchannels:int, expansion_mode:str,dropout_rate:float=0.01) -> None:
        super(Decoder,self).__init__()
        self.expansion_mode:str = expansion_mode
        
        self.conv:torch.nn.Sequential = torch.nn.Sequential(
            # ( bs, C, H, W ) >> (bs, C, H, W )
            torch.nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(outchannels),
            torch.nn.Dropout2d(dropout_rate),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(outchannels,outchannels,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(outchannels),
            torch.nn.Dropout2d(dropout_rate),
            torch.nn.ReLU(inplace=True),
        )

        if self.expansion_mode=='upsample':
            # (bs, C, H, W)  >> (bs, C//2 , H*2, W*2)
            self.expansion = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2,mode='nearest'),
                torch.nn.Conv2d(inchannels,outchannels,kernel_size=1,stride=1)
            )
        elif self.expansion_mode=='transponse':
            # (bs, C, H, W)  >> (bs, C//2 , H*2, W*2)
            self.expansion = torch.nn.ConvTranspose2d( inchannels, outchannels, kernel_size=2, stride=2 )
            
    def forward(self,x:torch.Tensor, skip:torch.Tensor)->torch.Tensor:
        '''
            refers (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

            x   :: (bs,16,16,512)
            skip:: (bs,32,32,256)
            x = upsample(x)              >> (bs,32,32,256)
            x = concat(x,skip)           >> (bs,32,32,256)+(bs,32,32,256) = (bs,32,32,512)
            x = double_conv(x)           >> (bs,32,32,512)
        '''
        x = self.expansion(x)
        # concat with x with skip on channel-wise
        x  = torch.cat((x,skip),dim=1)
        return self.conv(x)
        


class UNet(torch.nn.Module):
    def __init__(self,inchannels,outchannels,expansion_mode,contraction_mode,channels_list:list=[64,128,256,512], *args, **kwargs) -> None:
        super(UNet,self     ).__init__(*args, **kwargs)
        
        # Model
        self.expansion_mode:str = expansion_mode
        self.contraction_mode:str = contraction_mode

        # Encoder
        self.encoder1:Encoder = Encoder( inchannels, channels_list[0], self.contraction_mode,is_downsample=True)
        self.encoder2:Encoder = Encoder( channels_list[0], channels_list[1], self.contraction_mode,is_downsample=True)
        self.encoder3:Encoder = Encoder( channels_list[1], channels_list[2], self.contraction_mode,is_downsample=True)
        self.encoder4:Encoder = Encoder( channels_list[2], channels_list[3], self.contraction_mode,is_downsample=True)
        
        # Decoder
        self.decoder1:Decoder  = Decoder(channels_list[3],channels_list[2],self.expansion_mode)
        self.decoder2:Decoder  = Decoder(channels_list[2], channels_list[1],self.expansion_mode)
        self.decoder3:Decoder  = Decoder(channels_list[1],channels_list[0],expansion_mode=self.expansion_mode)
        
        # Final conv
        self.final_conv:torch.nn.Conv2d = torch.nn.Conv2d(channels_list[0],outchannels,kernel_size=1)
    def forward(self,x:torch.Tensor):
        '''
            channel_list=[32,64,128,512,1024]

            img:: ( bs, in, 256, 256 )

            ### Encoder Part
            - encoder1(bs, in, 256, 256)    >>  x1::( bs, 32,   128, 128 ), skip1::(bs, 32,  256, 256 )
            - encoder2(bs, 32, 128, 128 )   >>  x2::( bs, 64,   64,  64  ), skip2::(bs, 64 , 128, 128 )
            - encoder3(bs, 64, 64,  64 )    >>  x3::( bs, 128,  32,  32  ), skip3::(bs, 128, 64,  64  )
            - encoder4(bs, 128, 32, 32 )    >>  x4::( bs, 512,  16,  16  ), skip4::(bs, 512, 32,  32  )
            - encoder5(bs, 512, 16, 16 )    >>  x5::( bs, 1024, 16,  16  ), skip5::(bs, 1024, 16,  16 )

            ### Decoder Part
            - decoder1::( (bs,1024,16,16),(bs,512,32,32  ) ) >> ( (bs,512,32,32) + (bs,512,32,32) ) >> (bs,512,32,32)
            - decoder2::( (bs,512, 32,32),(bs,128,64,64  ) ) >> ( (bs,128,64,64) + (bs,128,64,64) ) >> (bs,128,64,64)
            - decoder3::( (bs,128, 64,64),(bs,64,128,128 ) ) >> ( (bs,64,128,128)+(bs,64,128,128) ) >> (bs,64,128,128)
            - decoder4::( (bs,64,128,128),(bs,32,256,256 ) ) >> ( (bs,32,256,256)+(bs,32,256,256) ) >> (bs,32,256,256)

            ### Final Conv Part
            - final_conv::(bs,32,256,256) >>> (bs,out,256,256)
        '''
        # Encoder Part
        x,skip1 = self.encoder1(x)
        x,skip2 = self.encoder2(x)
        x,skip3 = self.encoder3(x)
        _,x    = self.encoder4(x)

        # Decoder part
        x = self.decoder1(x,skip3)
        x = self.decoder2(x,skip2)
        x = self.decoder3(x,skip1)
        return self.final_conv(x)
    
class LitUNet(pl.LightningModule):
    def __init__(
            self, 
            inchannels:int,
            outchannels:int,
            expansion_mode:str,
            contraction_mode:str,
            channels_list:list[int,int,int,int,int],
            max_lr:float=1e-3,
            lr:float=1e-3,
            loss_fn:str = 'bce',
            *args: torch.Any, 
            **kwargs: torch.Any
        ) -> None:
        super().__init__(*args, **kwargs)

        
        assert( 
            (expansion_mode in ['upsample','transponse']) 
            and  (contraction_mode in ['strided_conv','maxpool'] ) 
            and  (len(channels_list)==4) 
        ), "expansion_mode::['upsample','transpose'] and contraction_mode::['strided_conv','maxpool'] and length(channel_list)==5"

        # Utility
        self.lr:float = lr 
        self.max_lr:float  = max_lr
        self.loss_fn:str = loss_fn

        # Model
        self.net = UNet(inchannels=inchannels, outchannels=outchannels, expansion_mode=expansion_mode, contraction_mode=contraction_mode,channels_list=channels_list).to(self.device)
        
    def forward(self,x:torch.Tensor, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        return self.net(x)
        
    def _common_step(self,batch):
        x,y,ohls = batch
        x    = x.to(self.device)
        ohls = ohls.type(torch.FloatTensor).to(self.device)
        y_pred = self(x)

        if self.loss_fn=='dice':
            loss = DiceLoss()(y_pred,ohls) 
        elif self.loss_fn=='bce':
            loss = torch.nn.BCEWithLogitsLoss()(y_pred,ohls)
        else:
            loss = torch.nn.CrossEntropyLoss()(y_pred,ohls)
        return loss

    def configure_optimizers(self) ->dict[torch.optim.Optimizer,dict[torch.optim.lr_scheduler.OneCycleLR]]:
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr,eps=1e-9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            epochs=self.trainer.max_epochs,
            pct_start=.3,
            div_factor=10,
            three_phase=True,
            final_div_factor=10,
            anneal_strategy='linear'
        )
        return {
            'optimizer':optimizer,
            'lr_scheduler':{
                'scheduler':scheduler,
                'interval':'step',
                'frequency':1
            }
        }
    
    def training_step(self,batch,batch_idx, *args: torch.Any, **kwargs: torch.Any) ->torch.Tensor:
        loss = self._common_step(batch)
        self.log(name='train_loss',value=loss.item(),prog_bar=True,on_step=True,on_epoch=True)
        return loss
    
    def validation_step(self, batch,batch_idx,*args: torch.Any, **kwargs: torch.Any) -> torch.Tensor :
        loss = self._common_step(batch)
        self.log(name='val_loss',value=loss.item(),prog_bar=True,on_step=True,on_epoch=True)
        return loss
    

if __name__=="__main__":
    # assert UNet(3,3,'upsample','maxpool')(torch.randn(16,3,256,256)).shape==(16,3,256,256),"model needs re-correct"
    # print("Model Loaded Properly!")
    pass