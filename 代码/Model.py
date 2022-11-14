#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/9/9 11:45 上午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : Model.py
# @Software: PyCharm
# @remarks : 无
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.01)
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self,input):
        # input(batch_size,time_step,input_size)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(input,(h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out
