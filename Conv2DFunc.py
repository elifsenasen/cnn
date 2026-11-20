import torch
import torch.nn.functional as F
import torch.nn as nn

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size)          
        self.stride = stride                          
        self.padding = padding   
        k = in_channels * kernel_size[0] * kernel_size[1]
        self.kernel = nn.Parameter(torch.randn(k, out_channels))



    def forward(self, input):                          
        padding, stride = self.padding, self.stride   
        batch_size, in_channels, height, width = input.size()
        assert in_channels == self.in_channels, f'expected {self.in_channels} channels, got {in_channels}'

        oh = (height + 2 * padding - self.kernel_size[0]) // stride + 1
        ow = (width + 2 * padding - self.kernel_size[1]) // stride + 1

        x = F.unfold(input, self.kernel_size, stride=stride, padding=padding)   
        k = in_channels * self.kernel_size[0] * self.kernel_size[1]
        assert x.size() == (batch_size, k, oh * ow), f'expected {(batch_size, k, oh * ow)}, got {x.size()}' 
        Yp = x.transpose(1, 2) @ self.kernel

        Y = Yp.transpose(1, 2).reshape(batch_size, self.out_channels, oh, ow)
        assert Y.size() == (batch_size, self.out_channels, oh, ow)
        return Y