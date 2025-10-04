import torch
from torch import nn
import torch.nn.functional as F



class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, 
        kernel_size=1, stride=1, 
        padding=0, dilation=1, 
        bias=False
    )->None:
        super().__init__()
        self.sep_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 
                kernel_size, stride, padding, dilation, groups=in_channels, bias=bias
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.sep_conv(x)
        return x
 

class Network(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dropout_val = 0.1
        self.bias = False

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(self.dropout_val),
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(self.dropout_val),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(self.dropout_val)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,padding=1,bias=self.bias),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, kernel_size=3, stride=1,padding=2,dilation=2, bias=self.bias), 
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=1,bias=self.bias),
            nn.BatchNorm2d(16),
        )

        self.ant1  = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,dilation=1,padding=0,stride=2,bias=self.bias),  
            nn.BatchNorm2d(16) 
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.dropout_val),
        )
        self.c5  = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=2,bias=self.bias,dilation=2), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.dropout_val)
        )

        self.ant2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,dilation=1,padding=0,stride=2,bias=self.bias,groups=64),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),
            nn.BatchNorm2d(64),

        )

        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.dropout_val),
        )
        self.c7  = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,bias=self.bias),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,padding=2,bias=self.bias,dilation=2), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(self.dropout_val)
        )

        self.ant3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=10,kernel_size=3,dilation=1,padding=0,stride=2,bias=self.bias),
            nn.BatchNorm2d(10)
        )

        self.c8 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,bias=self.bias),
            nn.BatchNorm2d(10),
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.op = nn.Conv2d(in_channels=10,out_channels=10,kernel_size=1,bias=self.bias)


    def forward(self,x):
        x = self.pre(x)
        x = self.c1(x)
        x = x + self.c2(x)
        x = x + self.c3(x)
        x = self.ant1(x)

        x = self.c4(x)
        x = x+ self.c5(x)
        x = self.ant2(x)

        x = x+self.c6(x)
        x = self.c7(x)
        x = self.ant3(x)

        x = self.c8(x)
        x = self.gap(x)
        x = self.op(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)
        