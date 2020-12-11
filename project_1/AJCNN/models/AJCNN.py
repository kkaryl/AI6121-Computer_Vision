from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
import operator

__all__ = ['AJCNN']

AJCNN_cfgs = {
  "AJCNN4": [[32, 32, 'M', 64, 64, 'M'],['D', 512, 'D', 512, 'D']],
  "AJCNN6": [[64, 64, 128, 'M', 128, 192, 'M', 192, 'M'],['D', 512, 'D', 256, 'D']],
  "AJCNN8": [[32, 32, 64, 'M', 64, 128, 128, 'M', 192, 192, 'M'],['D', 128, 'D', 128, 'D']],
  "AJCNN10": [[32, 32, 64, 'M', 64, 128, 128, 'M', 192, 192, 256, 'M', 256],['D', 128, 'D', 128, 'D']],
  "AJCNNT": [[32, 32, 64, 'M', 64, 128, 128, 'M', 192, 192, 'M'],['D', 64, 'D']]

}

def make_conv_layers(cfg, in_channels, batch_norm=True):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v) == int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_linear_layers(cfg, in_feats, out_feats, batch_norm=True):
    layers = []
    in_feats = in_feats
    for v in cfg:
        if v == 'D':
            layers += [nn.Dropout(p=0.5)]
        elif type(v) == int:
            linear = nn.Linear(in_feats, v)
            if batch_norm:
                layers += [linear, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [linear, nn.ReLU(inplace=True)]
            in_feats = v
    layers += [nn.Linear(in_feats, out_feats)]
    return nn.Sequential(*layers)

class AJCNN(nn.Module):
    """
    Adjustable CNN

    Inspired by VGG architecture. 
    Creates an adjustable CONV feature block from first array in configuration.
    Creates an adjustable Linear classifier block from second array in configuration.

    Automatically calculate flattened output from CONV feature block as inputs 
      to Linear classifier block.

    CONV feature block appends convolutional layers in the following configurable manner.
      If numerical: Adds a Conv2D layer, followed by optional BatchNorm2d, and ReLU
      If 'M': Adds a MaxPooling layer
    
    Linear classifer block appends linear layers in the following configurable manner:
      If numerical: Adds a linear layer, followed by optional BatchNom1d, and ReLU
      If 'D': Adds a Dropout layer with p=0.5
    """
    def __init__(self, in_channels: int = 1, 
        n_classes: int = 10,
        input_dim = (1,28,28), 
        variant:str = 'AJCNN8'):
        super(AJCNN, self).__init__()
        self.name = variant
          
        self.features = make_conv_layers(AJCNN_cfgs[variant][0], in_channels)
          
        num_feats_aft_conv = functools.reduce(operator.mul, list(self.features(torch.rand(1, *input_dim)).shape))

        self.classifier = make_linear_layers(AJCNN_cfgs[variant][1], num_feats_aft_conv, n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x     