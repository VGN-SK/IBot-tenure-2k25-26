import torch
import torch.nn as nn
import numpy as np

class SimpleNN(nn.Module) :

    def __init__(self,no_of_classes = 10) :
        super(SimpleNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()

        self.fcl1 = nn.Linear(128 * 3 * 3, 512)
        self.fcl2 = nn.Linear(512, no_of_classes)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x) :
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        #x = x.view(-1, 128 * 3 * 3)
        x = x.reshape(x.size(0), -1)


        x = self.fcl1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fcl2(x)

        return x