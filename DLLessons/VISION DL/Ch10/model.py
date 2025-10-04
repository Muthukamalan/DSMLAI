import torch
from torch import nn 
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self,input_channel,output_channel,stride=1, *args, **kwargs) -> None:
        super().__init__()
        self.conv1 =  nn.Sequential(
                            nn.Conv2d(
                                    in_channels = input_channel, 
                                    out_channels= output_channel,
                                    kernel_size=3,stride=stride,padding=1,bias=False                                 ),
                            nn.BatchNorm2d(output_channel),
                            nn.ReLU()
        )
        self.conv2  = nn.Sequential(
                            nn.Conv2d(
                                    in_channels=input_channel,
                                    out_channels=output_channel,
                                    kernel_size=3,stride=stride,padding=1,bias=False),
                            nn.BatchNorm2d(output_channel)
        )

    def forward(self,x):
        skip = x
        x = self.conv2(self.conv1(x))
        return F.relu(x+skip)


class CustomBlock(nn.Module):
    def __init__(self, input_channel, output_channel,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.intermediate = nn.Sequential(
            nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=3,bias=False,stride=1,padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        self.resblock  = BasicBlock(input_channel=output_channel, output_channel=output_channel)

    def forward(self,x):
        x = self.intermediate(x)
        rs = self.resblock(x)
        return x+rs



class ResNet(nn.Module):
    def __init__(self, dropout_val:float=0.01,bias:bool=False,*args, **kwargs) -> None:
        super(ResNet,self).__init__(*args, **kwargs)

        self.drop = dropout_val
        self.bias = bias
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,bias=self.bias,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    
        self.layer1 = CustomBlock(input_channel=64,output_channel=128)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=self.bias),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer3 = CustomBlock(input_channel=256,output_channel=512)

        self.maxpool = nn.MaxPool2d(kernel_size=4)
        self.fc      = nn.Linear(512,10)

    def forward(self,x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)
    