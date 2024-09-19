# 
# mibminet.py
# 
# Author(s):
# Xiaying Wang <xiaywang@iis.ee.ethz.ch>
# Lan Mei <lanmei@student.ethz.ch>
#
# Copyright (c) 2024 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch as t
import torch.nn.functional as F
import torch.utils.data as tud
import random
import numpy as np
import pandas as pd


class MIBMINet(t.nn.Module):

     
    def __init__(self, F1=8, D=2, F2=None, C=8, T=2000, N=4, Nf=64, Nf2=16, p_dropout=0.5, activation='relu', seed=-1,
                 dropout_type='dropout', pretrained=None, fold_id=0):
        """
        F1:           Number of spectral filters
        D:            Number of spacial filters (per spectral filter), F2 = F1 * D
        F2:           Number or None. If None, then F2 = F1 * D
        C:            Number of EEG channels
        T:            Number of time samples
        N:            Number of classes
	    Nf:           Size of temporal filter
        p_dropout:    Dropout Probability
        activation:   string, either 'elu' or 'relu'
        dropout_type: string, either 'dropout', 'SpatialDropout2d' or 'TimeDropout2D'
        """
        super(MIBMINet, self).__init__()

        # prepare network constants
        if F2 is None:
            F2 = F1 * D

        # check the activation input
        activation = activation.lower()
        assert activation in ['elu', 'relu']
        print("activation relu?: ", activation)

        # Prepare Dropout Type
        if dropout_type.lower() == 'dropout':
            dropout = t.nn.Dropout
        elif dropout_type.lower() == 'spatialdropout2d':
            dropout = t.nn.Dropout2d
        elif dropout_type.lower() == 'timedropout2d':
            dropout = TimeDropout2d
        else:
            raise ValueError("dropout_type must be one of SpatialDropout2d, Dropout or "
                             "WrongDropout2d")

        # store local values
        self.F1, self.D, self.F2, self.C, self.T, self.N = (F1, D, F2, C, T, N)
        self.p_dropout, self.activation = (p_dropout, activation)

        # Number of input neurons to the final fully connected layer
        n_features = (T // 8) // 8 #(T // 4) // 8 #23 #(T // 8) // 8 

        # Block 1
        self.conv1 = t.nn.Conv2d(1, F1, (C, 1), bias=False)
        self.batch_norm1 = t.nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.before_conv2_pad = t.nn.ZeroPad2d((Nf//2-1, Nf//2, 0, 0)) 
        self.conv2 = t.nn.Conv2d(F1, D * F1, (1, Nf), groups=F1, bias=False)
        self.batch_norm2 = t.nn.BatchNorm2d(D * F1, momentum=0.01, eps=0.001)
        self.activation1 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool1 = t.nn.AvgPool2d((1, 8)) # 4 # 8
        # self.dropout1 = dropout(p=p_dropout)
        self.dropout1 = dropout(p=p_dropout) # t.nn.Dropout(p=p_dropout)

        # Block 2
        #self.sep_conv_pad = t.nn.ZeroPad2d((7, 8, 0, 0))
        self.sep_conv_pad = t.nn.ZeroPad2d((Nf2//2-1, Nf2//2, 0, 0)) 
        self.sep_conv1 = t.nn.Conv2d(D * F1, D * F1, (1, Nf2), groups=D * F1, bias=False)
        self.sep_conv2 = t.nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.batch_norm3 = t.nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = t.nn.ELU(inplace=True) if activation == 'elu' else t.nn.ReLU(inplace=True)
        self.pool2 = t.nn.AvgPool2d((1, 8))
        self.dropout2 = dropout(p=p_dropout)

        # Fully connected layer (classifier)
        self.flatten = t.nn.Flatten()
        self.fc = t.nn.Linear(F2 * n_features, N, bias=True) # input size: F2*n_features, output size: N

        # initialize weights
        self._initialize_params(seed=seed)

        print("fold_id: ")
        print(fold_id)

        if pretrained is not None:
            if isinstance(pretrained, list):
                #self.load_state_dict(torch.load(pretrained))
                ckpt = torch.load(pretrained[fold_id])
            else:
                ckpt = torch.load(pretrained)
            # network parameters
            self.load_state_dict(ckpt['net'])


    def forward(self, x):

        # input dimensions: (s, 1, C, T) [batchsize, 1, numb of Channels, T]

        # Block 1
        
        x = self.conv1(x)            # output dim: (s, F1, C, T-1) [batchsize, F1, 1, T]
        x = self.batch_norm1(x)      #[batchsize, F1, 1, T]
        x = self.before_conv2_pad(x)  
        x = self.conv2(x)            
        
        x = self.batch_norm2(x)
        x = self.activation1(x)
        x = self.pool1(x)            # output dim: (s, D * F1, 1, T // 8)
        x = self.dropout1(x)

        # Block2
        x = self.sep_conv_pad(x)
        x = self.sep_conv1(x)        # output dim: (s, D * F1, 1, T // 8 - 1)
        x = self.sep_conv2(x)        # output dim: (s, F2, 1, T // 8 - 1)
        
        x = self.batch_norm3(x)
        x = self.activation2(x)
        x = self.pool2(x)            # output dim: (s, F2, 1, T // 64)
        x = self.dropout2(x)
        

        # Classification
        x = self.flatten(x)          # output dim: (s, F2 * (T // 64))
        x = self.fc(x)               # output dim: (s, N)
        
        # if with_stats:
        #     stats = [('conv1_w', self.conv1.weight.data),
        #             ('conv2_w', self.conv2.weight.data),
        #             ('sep_conv1_w', self.sep_conv1.weight.data),
        #             ('sep_conv2_w', self.sep_conv2.weight.data),
        #             ('fc_w', self.fc.weight.data),
        #             ('fc_b', self.fc.bias.data)]
        #    return stats, x
        
        return x

    def forward_with_tensor_stats(self, x):
        return self.forward(x)

    def _initialize_params(self, seed, weight_init=t.nn.init.xavier_uniform_, bias_init=t.nn.init.zeros_):
        """
        Initializes all the parameters of the model

        Parameters:
         - weight_init: t.nn.init inplace function
         - bias_init:   t.nn.init inplace function

        """
        
        if seed >= 0:
            torch.manual_seed(seed)
            # using GPU
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        def init_weight(m):
            if isinstance(m, t.nn.Conv2d) or isinstance(m, t.nn.Linear):
                weight_init(m.weight)
            if isinstance(m, t.nn.Linear):
                bias_init(m.bias)

        self.apply(init_weight)


class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class TimeDropout2d(t.nn.Dropout2d):
    """
    Dropout layer, where the last dimension is treated as channels
    """
    def __init__(self, p=0.5, inplace=False):
        """
        See t.nn.Dropout2d for parameters
        """
        super(TimeDropout2d, self).__init__(p=p, inplace=inplace)

    def forward(self, input):
        if self.training:
            input = input.permute(0, 3, 1, 2)
            input = F.dropout2d(input, self.p, True, self.inplace)
            input = input.permute(0, 2, 3, 1)
        return input

