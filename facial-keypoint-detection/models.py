## TODO: define the convolutional neural network architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 4)  # batch, 221, 221, 32
        self.conv2 = nn.Conv2d(32, 64, 3)  # batch, 108, 108, 64
        self.conv3 = nn.Conv2d(64, 128, 2)  # batch, 53, 53, 128
        self.conv4 = nn.Conv2d(128, 256, 1)  # batch, 26, 26, 256
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1000)
        self.bn6 = nn.BatchNorm1d(1000)
        
        
        self.pool1 = nn.MaxPool2d(2)  # batch, 110, 110, 32 
        self.pool2 = nn.MaxPool2d(2)  # batch, 54, 54, 64
        self.pool3 = nn.MaxPool2d(2)  # batch, 26, 26, 128
        self.pool4 = nn.MaxPool2d(2)  # batch, 13, 13, 256
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

        self.fc1 = nn.Linear(43264, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        

        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)
        nn.init.xavier_uniform_(self.fc3.weight.data)
        
    def forward(self, x):
        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)

        x = self.dropout5(F.elu(self.bn5(self.fc1(x))))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x
