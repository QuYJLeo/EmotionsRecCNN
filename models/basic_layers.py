import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):
    """A residual block for residual networks.
    
    This block follows the bottleneck design pattern with 1x1, 3x3, and 1x1 convolutions.
    It includes batch normalization and ReLU activation layers.

    Args:
        input_channels (int): Number of input channels
        output_channels (int): Number of output channels
        stride (int): Stride for the convolution layers, default is 1
    """
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels, int(output_channels / 4), 1, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            stride,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            int(output_channels / 4), output_channels, 1, 1, bias=False
        )
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        """Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor after residual connection
        """
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out
