#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self,size):
        super(Highway,self).__init__()
        self.Linear1=nn.Linear(in_features=size,out_features=size,bias=True)
        self.Linear2=nn.Linear(in_features=size,out_features=size,bias=True)
    def forward(self,x_conv_out):
        x_proj=self.Linear1(x_conv_out)
        x_proj=F.relu(x_proj)
        x_gate=self.Linear2(x_conv_out)
        x_gate=torch.sigmoid(x_gate)
        x_highway=x_gate*x_proj+(1-x_gate)*x_conv_out
        return x_highway
        

### END YOUR CODE 
