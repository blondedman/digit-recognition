import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    
    def __int__(self, classes):
        
        # call to parent construct
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxp1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(800, 500)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(500, 10)
        self.logSoftmax = F.log_softmax(dim=1)
    
    def forward(self, x):	
        	
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxp1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxp2(x)
        
        x = nn.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        output = self.logSoftmax(x)
        
        return output

model = LeNet()
print(model)