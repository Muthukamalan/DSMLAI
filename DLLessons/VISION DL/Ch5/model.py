import torch
from utils import device 

#TODO: Our CNN Model
class Net(torch.nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = torch.nn.Linear(4096, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))                  # 28>26    | 1>3     | 1>1 
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x),2))  # 26>24>12 | 3>5>6   | 1>1>2
        x = torch.nn.functional.relu(self.conv3(x))                  # 12>10    | 6>10    | 2>2
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv4(x),2))  # 10>8>4   | 10>14>16| 2>2>4
        x = x.view(-1, 4096)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

model = model = Net().to(device)