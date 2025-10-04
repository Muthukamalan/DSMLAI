import torch
from torch import nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self,inc:int,outc:int,k:int,p:int,norm:str,dp_rate:int,grp:int=0):
        super(ConvLayer,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=3,padding=p,bias=False),
            self.get_norm(norm=norm,grp=grp,num_f=outc),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dp_rate)
        )

    def get_norm(self,norm:str,num_f:int,grp:int=0):
        if norm=='bn':
            return nn.BatchNorm2d(num_features=num_f)
        elif norm=='ln':
            return nn.GroupNorm(num_groups=1,num_channels=num_f)
        elif norm=='gn':
            return nn.GroupNorm(num_groups=grp,num_channels=num_f)
        else:
            raise ValueError("choose bn/ln/gn")

    def forward(self,x):
        x = self.layer(x)
        return x


class TransBlock(nn.Module):
    def __init__(self,inc:int,outc:int):
        super(TransBlock,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=inc,out_channels=outc,kernel_size=1,bias=False),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
    def forward(self,x):
        x = self.layer(x)
        return x
    

class Net(nn.Module):
        def __init__(self,norm_method:str,channels:list=[3,8,12,20,32,10],grp:int=0,dp_rate=0.1):
            super(Net,self).__init__()
            if not norm_method in('bn','ln','gn'):
                raise ValueError('choose bn/ln/gn')
            else:
                self.norm = norm_method
                if self.norm =='gn':
                    self.grp=grp
                elif self.norm=='ln':
                    self.grp=1
                else:
                    self.grp=0

            self.dp_rate=dp_rate
            
            self.C1 = ConvLayer(inc=3,outc=channels[1],k=3,p=1,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)
            self.C2 = ConvLayer(inc=channels[1],outc=channels[1],k=3,p=1,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)
            self.t1 = TransBlock(inc=channels[1],outc=channels[1])
            
            self.C4 = ConvLayer(inc=channels[1],outc=channels[2],k=3,p=1,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)
            self.C5 = ConvLayer(inc=channels[2],outc=channels[2],k=3,p=1,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)
            self.t2 = TransBlock(inc=channels[2],outc=channels[3])

            self.C7 = ConvLayer(inc=channels[3],outc=channels[4],p=1,k=3,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)
            self.C8 = ConvLayer(inc=channels[4],outc=channels[4],p=1,k=3,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)
            self.C9 = ConvLayer(inc=channels[4],outc=channels[4],p=1,k=3,norm=self.norm,dp_rate=self.dp_rate,grp=self.grp)

            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.out = nn.Conv2d(in_channels=channels[4],out_channels=channels[-1],kernel_size=1,stride=1,bias=False)
    
        def forward(self,x):
            x = self.C1(x)
            x = x + self.C2(x)
            x = self.t1(x)

            x = self.C4(x)
            x = x + self.C5(x)
            x = self.t2(x)

            x = self.C7(x)
            x = x + self.C8(x)
            x = x + self.C9(x)
            x = self.gap(x)
            x = self.out(x)
            return F.log_softmax(x.view(-1,10), dim=1)
        



class Network(nn.Module):
    def __init__(self, method='batch',*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dropout_val = 0.01
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,16),
            nn.Dropout2d(self.dropout_val),


            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,16),
            nn.Dropout2d(self.dropout_val),


            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=1,padding=0,bias=self.bias)
        )

        self.pool1  = nn.MaxPool2d(2,2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,16),
            nn.Dropout2d(self.dropout_val),


            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,16),
            nn.Dropout2d(self.dropout_val),

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,32),
            nn.Dropout2d(self.dropout_val),


            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,padding=0,bias=self.bias)
        )

        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,32),
            nn.Dropout2d(self.dropout_val),

            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,32),
            nn.Dropout2d(self.dropout_val),

            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            self.get_norm_layer(method,32),
            nn.Dropout2d(self.dropout_val)
        )

        self.gap = nn.AvgPool2d(kernel_size=8)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=10,kernel_size=1,padding=0,bias=self.bias)
        )

    
    def get_norm_layer(self,method:str,output_channels:int)->nn.Module:
        if method=='batch':
            return nn.BatchNorm2d(output_channels)
        elif method=='group':
            return nn.GroupNorm(4,output_channels)
        elif method=='layer':
            return nn.GroupNorm(1,output_channels)
        
    def forward(self,x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.gap(self.conv3(x))
        x = self.conv4(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)