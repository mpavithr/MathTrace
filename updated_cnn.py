import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Net(nn.Module):

    def __init__(self, input_shape, vocab, activation_function='relu', pooling_type='avg',):
        super().__init__()
        self.n_classes = len(vocab)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv1_bn = nn.BatchNorm2d(6)
        if pooling_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        elif pooling_type == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv2_bn = nn.BatchNorm2d(16)
        if pooling_type == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        elif pooling_type == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(115200, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, self.n_classes)
        if activation_function == 'sigmoid':
            self.activation_function = F.sigmoid
        elif activation_function == 'tanh':
            self.activation_function = F.tanh
        elif activation_function == 'relu':
            self.activation_function = F.relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.activation_function(self.conv1_bn(x)))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool(self.activation_function(self.conv2_bn(x)))
        x = x.flatten(start_dim=1)   # flatten features
#         print(x.shape)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.activation_function(self.fc3(x))
        x = self.fc4(x)
        return F.sigmoid(x)

    def decode(self, x):
        return x
