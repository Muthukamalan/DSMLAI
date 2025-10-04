from init import * 
from utils import config


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        DROPOUT= config['model']['dropout_rate']
        BIAS   = config['model']['bias']
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1,bias=BIAS),
            # nn.BatchNorm2d(3),
            # nn.ReLU(),
            # nn.Dropout2d(p=DROPOUT),

            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=DROPOUT),
            
            nn.Conv2d(in_channels=8,out_channels=10,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=DROPOUT),

            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=DROPOUT),
        )
        self.trans1 = nn.Sequential(

            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =1 ),
            nn.Conv2d(in_channels=10,out_channels=8,kernel_size=1,bias=BIAS,padding=1),
        )

        self.conv2 =nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=10,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),


            nn.Conv2d(in_channels=10,out_channels=12,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),

            nn.Conv2d(in_channels=12,out_channels=12,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),
        )
        self.trans2 = nn.Sequential(
            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =1 ),
            nn.Conv2d(in_channels=12,out_channels=8,kernel_size=1,bias=BIAS),
            nn.BatchNorm2d(8),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=10,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(p=DROPOUT),
            
            nn.Conv2d(in_channels=10,out_channels=12,kernel_size=3,stride=1,padding=1,bias=BIAS),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=DROPOUT),

        )
        self.trans3 = nn.Sequential(
            nn.Conv2d(in_channels=12,out_channels=10,kernel_size=1,bias=BIAS),
            nn.MaxPool2d( kernel_size =2 , stride =2 , padding =0 ),
            nn.BatchNorm2d(10),
        )

        self.out4 = nn.Sequential(
            nn.Conv2d(in_channels=10 ,out_channels=10, kernel_size=3,stride=1,padding=1,bias=BIAS),
            # nn.BatchNorm2d(10),
            nn.AvgPool2d(kernel_size=3)  #(1*1*10)
        )


    def forward(self,x):
        x = self.trans1( self.conv1(x) )
        x = self.trans2( self.conv2(x) )
        x = self.trans3( self.conv3(x) )
        x = self.out4(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)





class Network(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Network,self).__init__()
        dropout = 0.04
        bias    = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding=0,bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,padding=0,bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout2d(dropout)
        )

        self.trans1 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=10,kernel_size=1,padding=0,bias=bias),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,padding=0,bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=0,bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=bias),
            nn.ReLU(),            
            nn.BatchNorm2d(20),
            nn.Dropout(dropout)
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(20)
        )

        self.conv4 = nn.Sequential(
            # nn.Conv2d(in_channels= 20, out_channels= 22, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
        ) 

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.conv4(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)
    



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )  



        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )



        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 



        self.pool1 = nn.MaxPool2d(2, 2) 



        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) 



        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) 



        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) 



        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )



        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        ) 



        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(2, 2), padding=0, bias=False),
        ) 



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
