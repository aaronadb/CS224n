#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CNN,self).__init__()
        self.conv=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=5)
        self.maxpool=nn.AdaptiveMaxPool1d(output_size=1)
    def forward(self,x_reshaped):
        x_conv=self.conv(x_reshaped)
        x_conv_out=F.relu(x_conv)
        x_conv_out=self.maxpool(x_conv_out)
        x_conv_out=torch.squeeze(x_conv_out,dim=-1)
        return x_conv_out
        

### END YOUR CODE
