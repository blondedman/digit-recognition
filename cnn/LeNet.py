import torch
import torchvision

import torch.nn as nn

class LeNet(nn.Module):
    
    def __init__(self):
        
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(1, 20, 5),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2, 2),    
        )
        
        self.conv2 = nn.Sequential(         
            nn.Conv2d(20, 50, 5),     
            nn.ReLU(),                      
            nn.MaxPool2d(2, 2),                
        )
        
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.rel = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=10)
        
        self.out = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.rel(x)
        x = self.fc2(x)
        
        output = self.out(x)
        return output

model = LeNet()
print(model)