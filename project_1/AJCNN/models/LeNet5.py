from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


__all__ = ['LeNet5']


class LeNet5(nn.Module):

    def __init__(self, kernel=5, pad=2, 
                 activation='sigmoid', pool='avg', 
                 num_filter1=6, num_filter2=16, linear1=400):
        super(LeNet5, self).__init__()
        self.name = 'LeNet5'
        self.conv1 = nn.Conv2d(1,   num_filter1,  kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv2d(num_filter1,   num_filter2,  kernel_size=kernel)  
        self.linear1 = nn.Linear(linear1, 120) 
        self.linear2 = nn.Linear(120, 84) 
        self.linear3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.001)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        if pool == 'avg':
            self.pool = nn.AvgPool2d(2, 2)
        elif pool == 'max':
            self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)  
        x = self.activation(x) 
        x = self.pool(x) 
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) 
        x = self.linear1(x) 
        x = F.relu(x) 
        x = self.dropout(x)
        x = self.linear2(x) 
        x = F.relu(x) 
        x = self.dropout(x)
        x = self.linear3(x)
    
        return x

