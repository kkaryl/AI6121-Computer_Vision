from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImageRNN(nn.Module):
    def __init__(self):
        super(ImageRNN, self).__init__()
        self.name = 'ImageRNN'
        
        self.hidden_size = 256
        self.batch_size = 128
        self.input_size = 28
        self.n_classes = 10
        self.device = None
        
        self.basic_rnn = nn.RNN(self.input_size, self.hidden_size) 
        
        self.FC = nn.Linear(self.hidden_size, self.n_classes)
        
    def init_hidden(self):
        # (num_layers, batch_size, hidden_size)
        return (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))

    def init_cell(self):
        return (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))
        
    def forward(self, X):
        # print(X.shape)
        X = X.squeeze(1).permute(1, 0, 2) 
        self.device = X.get_device()
        # print(X.shape)

        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()
        self.cell = self.init_cell()
        
        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)      
        out = self.FC(self.hidden)
        
        return out.view(-1, self.n_classes) 








