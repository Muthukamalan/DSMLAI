import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        DROPOUT=0.01
        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),

            nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),
            
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),
        )
        self.trans1 = nn.Sequential(

            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =1 ),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,bias=False,padding=1),
            nn.BatchNorm2d(8),

        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),

            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),

            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),
        )
        self.trans2 = nn.Sequential(
            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =1 ),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=1,bias=False),
            nn.BatchNorm2d(16),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),
            
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=DROPOUT),

        )
        self.trans3 = nn.Sequential(
            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =0 ),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=1,bias=False),
            nn.BatchNorm2d(16),
        )

        self.out4 = nn.Sequential(
            nn.Conv2d(in_channels=16 ,out_channels=10, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(10),
            nn.AvgPool2d(kernel_size=3)  #(1*1*10)
        )


    def forward(self,x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.trans3( self.conv3(x) )
        x = self.out4(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)
