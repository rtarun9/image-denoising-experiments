
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')
from dataloader import get_train_and_validation_dataloader

class InputProjectionLayer(torch.nn.Module):
    def __init__(self, num_channels, kernel_size=(3,3)):
        super(InputProjectionLayer, self).__init__()
        
        self.conv = torch.nn.Conv2d(1, num_channels, kernel_size, padding='same')
        
    def forward(self, images):
        images = torch.transpose(images, 3, 1)
        return self.conv(images)
    

class OutputProjectionLayer(torch.nn.Module):
    def __init__(self, num_channels, kernel_size=(3,3)):
        super(OutputProjectionLayer, self).__init__()
        
        self.conv2d_transpose = torch.nn.ConvTranspose2d(num_channels, 1, kernel_size, padding=1)
        
    def forward(self, images):
        images = self.conv2d_transpose(images)
        images = torch.transpose(images, 3,  1)
        return images
    
class DepthWiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                               groups=in_channels, bias=bias, padding='same')
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, num_channels, width, height):
        super(ConvBlock, self).__init__()
        
        self.zero_padding = torch.nn.ZeroPad2d(padding=3)
        self.dwconv = DepthWiseSeparableConv2d(num_channels, num_channels, 7)
        self.layer_norm = torch.nn.LayerNorm([1, num_channels, width + 6, height + 6], elementwise_affine=False)
        self.linear_1 = torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=7)
        self.gelu = torch.nn.GELU()
        self.linear_2 = torch.nn.Conv2d(num_channels * 4, num_channels, kernel_size=1) 
        
    def forward(self, images):
        residual = images
        
        images = self.zero_padding(images)
        images = self.dwconv(images)
        images = self.layer_norm(images)
        images = self.linear_1(images)
        images = self.gelu(images)
        images = self.linear_2(images)
        return images + residual 

            
import torch
import torch.nn as nn
import torch.nn.functional as F

class YBlock(nn.Module):
    def __init__(self, attention_scaling_factor, num_channels, width, height):
        super(YBlock, self).__init__()

        self.dk = attention_scaling_factor
        self.c = num_channels

        # Defining the layers for path B (self-attention, conv)

        # (Q, K, V) are used for attention.
        # This is clear from the paper which mentions:
        # The (original) self-attentive mechanism first generates the corresponding query e
        # (q), Key (K) and Value (V)
        # The output of K and Q linear are: HW/k^2 x C
        # Note that the vectors (output of self.k/q/v_linear) are PER channel (C).
        self.k_linear = nn.Linear(self.c, self.c, bias=True)
        self.q_linear = nn.Linear(self.c, self.c, bias=True)
        self.v_linear = nn.Linear(self.c, self.c, bias=True)
        self.self_attention_linear = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        # Defining the layers for the stage where the input is the output of path A and path B
        self.norm = nn.LayerNorm([num_channels, height, width], elementwise_affine=False)  
        self.linear_1 = nn.Conv2d(num_channels, num_channels * 4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=1)

    def forward(self, images):
        images_dup = images

        # Attention module computation.
        # Q, K, V all pass through linear layers.

        # We are converting that into (num_images, height' * width', channel)
        flattened_images = images.view(images.size(0), -1, self.c)

        q = self.q_linear(flattened_images)
        # For the computation of the attention map, we need the shape of q to be num_images, num_channels, HW.
        # But now, it is num_images, HW, num_channels. The last 2 dimensions must be reversed.
        q = q.permute(0, 2, 1)

        k = self.k_linear(flattened_images)

        flattened_images = images.view(images.size(0), -1, self.c)

        v = self.v_linear(flattened_images)

        # Computation of attention
        # attention = ((K'T Q) / sqrt(dk)) V
        # Ignoring num_images, the shape of Q is (num_channels)
        attention = torch.matmul(q, k)

        # As per the paper, after performing softmax, we obtain an attention score matrix with dimension C x C.
        # The scaling factor mentioned in the paper (i.e 1/sqrt(dk)) is based on network depth.
        attention = F.softmax(attention / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32)), dim=-1)

        # Now, the final attention map is obtained by multiplying v and attention.
        attention = torch.matmul(v, attention)


        # Now, attention is reshaped into the same dimensions as the input image.
        sa_output = attention.view(-1, images_dup.size(1), images_dup.size(2), images_dup.size(3))

        output = images_dup + sa_output 

        return output 


import ptwt
import pywt 

class YModel(torch.nn.Module):
    def __init__(self, num_channels, width=512, height=512):
        super(YModel, self).__init__()
        
        self.ip = InputProjectionLayer(num_channels=num_channels)
        self.op = OutputProjectionLayer(num_channels=num_channels)

        # Pass A : Downsampling passes
        self.conv_block_1 = ConvBlock(num_channels=num_channels, width=width, height=height)
       
        self.y_block_1 = YBlock(attention_scaling_factor=1, num_channels=num_channels, width=width, height=height)

        self.downsampling_layer_1 = torch.nn.Conv2d(num_channels, num_channels * 2, kernel_size=(2,2), stride=2)

        self.conv_block_2 = ConvBlock(num_channels=num_channels * 2, width=width // 2, height=height // 2)
        
        self.y_block_2 = YBlock(attention_scaling_factor=3, num_channels=num_channels * 2, width=width // 2, height=height // 2)

        self.downsampling_layer_2 = torch.nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=(2,2), stride=2)
        
        self.conv_block_3 = ConvBlock(num_channels=num_channels * 4, width=width // 4, height=height // 4)

        self.y_block_3 = YBlock(attention_scaling_factor=5, num_channels=num_channels * 4, width=width // 4, height=height // 4)

        self.conv_block_4 = ConvBlock(num_channels=num_channels * 4, width=width // 4, height=height // 4) 

        # Pass B : Upsampling passes.
        self.upsampling_layer_1 = torch.nn.ConvTranspose2d(num_channels * 4, num_channels * 2,  kernel_size=(2, 2), stride=2)

        self.y_block_4 = YBlock(attention_scaling_factor=7, num_channels=num_channels * 2, width=width // 2, height=height // 2)
        self.conv_block_5 = ConvBlock(num_channels=num_channels * 2, width=width // 2, height=height // 2)

        self.upsampling_layer_2 = torch.nn.ConvTranspose2d(num_channels * 2, num_channels * 1,  kernel_size=(2, 2), stride=2)

        self.y_block_5 = YBlock(attention_scaling_factor=7, num_channels=num_channels * 2, width=width // 2, height=height // 2)
        self.conv_block_6 = ConvBlock(num_channels=num_channels, width=width , height=height )


    def forward(self, images):
        images = self.ip(images)

        cb1 = self.conv_block_1(images)

        down1 = self.downsampling_layer_1(cb1)

        cb2  = self.conv_block_2(down1)

        down2 = self.downsampling_layer_2(cb2)

        cb3 = self.conv_block_3(down2)
        y3 = self.y_block_3(cb3)
        cb4 = self.conv_block_4(y3)

        up1 = self.upsampling_layer_1(cb4)

        cb5 = self.conv_block_5(up1) + cb2

        up2 = self.upsampling_layer_2(cb5)

        cb6 = self.conv_block_6(up2) + cb1

        output = self.op(cb6)
        return output