import torch

import torchvision



import torch.nn as nn

import torch.nn.functional as F


class cnnMnist (nn.Module):
    def __init__ (self):
        super(cnnMnist, self).__init__()

        self.conv1 = nn.Conv2d(1,4,7)

        self.fc1 = nn.Linear(484,242)

        self.fc2 = nn.Linear(242, 121)

        self.fc3 = nn.Linear(121, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)),2)

        x = torch.flatten(x,1)
        
        x= F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x
