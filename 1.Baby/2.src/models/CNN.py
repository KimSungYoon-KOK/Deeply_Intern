import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple_CNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        self.fc1 = nn.Linear(32*14*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_class)

    def forward(self, x):
        # print("input : ", x.shape)                      # input :  torch.Size([16, 1, 64, 33])
        x = self.pool(F.relu(self.conv1(x)))
        # print("conv1 : ", x.shape)                      # conv1 :  torch.Size([16, 16, 31, 15])
        x = self.pool(F.relu(self.conv2(x)))
        # print("conv2 : ", x.shape)                      # conv2 :  torch.Size([16, 32, 14, 6])
        x = torch.flatten(x, start_dim=1)   
        # print("flatten : ", x.shape)                    # flatten :  torch.Size([16, 2688])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

"""Convolution - Batch Normalization - Activation - Dropout - Pooling"""

class CNN_v2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch3 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_class)

    def forward(self, x):
        # print("input : ", x.shape)                      # input :  torch.Size([16, 1, 64, 33])
        layer1 = self.conv1(x)
        layer1 = self.batch1(layer1)
        layer1 = self.pool(F.relu(layer1))
        # print("layer1 : ", layer1.shape)                # layer1 :  torch.Size([16, 16, 31, 15])

        layer2 = self.conv2(layer1)
        layer2 = self.batch2(layer2)
        layer2 = self.pool(F.relu(layer2))
        # print("layer2 : ", layer2.shape)                 # layer2 :  torch.Size([16, 32, 14, 6])

        layer3 = self.conv3(layer2)
        layer3 = self.batch3(layer3)
        layer3 = self.pool(F.relu(layer3))
        # print("layer3 : ", layer3.shape)                 # layer3 :  torch.Size([16, 64, 6, 2])

        flatten = torch.flatten(layer3, start_dim=1)
        # print("flatten : ", flatten.shape)               # flatten :  torch.Size([16, 768])

        fc = F.relu(self.fc1(flatten))
        fc = self.dropout(fc)
        fc = F.relu(self.fc2(fc))
        output = self.fc3(fc)

        return output

