import torch
import torchvision

import torch.nn as nn

class LeNet(nn.Module):
    
    def __int__(self, channels, classes):
        
        # call to parent construct
        super(LeNet, self).__init__
        
        self.conv1 = nn.Conv2d(channels, 20)
        self.relu1 = nn.ReLU()
        self.maxp1 = nn.MaxPool2d()
        
    pass