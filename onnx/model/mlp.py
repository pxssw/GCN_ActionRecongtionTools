import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation import activation_factory
import ipdb
from ptflops import get_model_complexity_info


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            # self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            #1*1 in_channels-16-32-out_channels
            self.layers.append(nn.Conv2d(channels[i-1], 16, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(16))
            self.layers.append(activation_factory(activation))
            # 16-32
            # self.layers.append(nn.Conv2d(16, 32, kernel_size=1))
            # self.layers.append(nn.BatchNorm2d(32))
            # self.layers.append(activation_factory(activation))
            # 32-64 out_channels
            self.layers.append(nn.Conv2d(16, channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation))


    def forward(self, x):
        # Input shape: (N,C,T,V)
        # ipdb.set_trace()
        for layer in self.layers:
            x = layer(x)
        return x


class MLP2(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            # self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            #1*1 in_channels-16-32-out_channels
            self.layers.append(nn.Conv2d(channels[i-1], 16, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(16))
            self.layers.append(activation_factory(activation))
            # 16-32
            # self.layers.append(nn.Conv2d(64, 128, kernel_size=1))
            # self.layers.append(nn.BatchNorm2d(128))
            # self.layers.append(activation_factory(activation))
            # 64-128 out_channels
            self.layers.append(nn.Conv2d(16, channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation))


    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # bugs cpu errors
    msgcn = MLP2(3 * 13, [128]).cuda()
    msgcn.forward(torch.randn(1,3*13,10,25).cuda())

    flops, params = get_model_complexity_info(msgcn, (3*13, 10, 25), as_strings=True, print_per_layer_stat=True)
    print("%s |%s |%s" % ('MSG3D', flops, params))
