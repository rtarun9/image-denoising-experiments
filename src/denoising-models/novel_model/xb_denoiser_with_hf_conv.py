import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')
from dataloader import get_train_and_validation_dataloader
import pywt, ptwt

class InputProjectionLayer(torch.nn.Module):
    def __init__(self):
        super(InputProjectionLayer, self).__init__()
        
    def forward(self, images):
        print(images.shape)
        images = torch.transpose(images, 3, 1)
        images = torch.squeeze(images)
        yl, yh = ptwt.wavedec2(images, pywt.Wavelet('db3'), level=1, mode="constant")
        
        return torch.unsqueeze(yl, 0), yh
    
class OutputProjectionLayer(torch.nn.Module):
    def __init__(self):
        super(OutputProjectionLayer, self).__init__()
        
    def forward(self, images):
        op = ptwt.waverec2((images[0], (images[1][0],images[1][1], images[1][2])), pywt.Wavelet('db3'))
        images = torch.transpose(op, 3,  1)
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

class XBBlock(nn.Module):
    def __init__(self, attention_scaling_factor, num_channels, width, height):
        super(XBBlock, self).__init__()

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

class XModel(torch.nn.Module):
    def __init__(self, num_channels, width=512, height=512):
        super(XModel, self).__init__()
        width, height = 258, 258
        
        self.upsampling_layer_1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(2, 2), stride=2)
        self.downsampling_layer_1 = torch.nn.Conv2d(num_channels, num_channels, kernel_size=(2,2), stride=2)


        self.ip = InputProjectionLayer()
        self.op = OutputProjectionLayer()

        # Low frequency network blocks.
        self.low_freq_conv_block_1 = ConvBlock(num_channels=1, width=width, height=height)
        self.low_freq_conv_block_2 = ConvBlock(num_channels=1, width=width, height=height) 
        

        self.hf1_conv_block_1 = ConvBlock(num_channels=1, width=width, height=height)
        self.hf2_conv_block_1 = ConvBlock(num_channels=1, width=width, height=height)
        self.hf3_conv_block_1 = ConvBlock(num_channels=1, width=width, height=height)

        self.xbblock_low_freq = XBBlock(attention_scaling_factor = 5, num_channels=1, width=129, height=129)

    def forward(self, images):
        low_freq, high_freq = self.ip(images)

        # Low freq pass.
        lf_conv_block_1 = self.low_freq_conv_block_1(low_freq)
        downsampled_1 = self.downsampling_layer_1(lf_conv_block_1)
        lf_xbb= self.xbblock_low_freq(downsampled_1)
        upsampled_1  = self.upsampling_layer_1(lf_xbb) + lf_conv_block_1
        lf_conv_block_2 = self.low_freq_conv_block_2(upsampled_1) +low_freq 

        # For high freq : Only convolution.
        hf = [None] * 3
        hf[0] = torch.unsqueeze(self.hf1_conv_block_1(torch.unsqueeze(high_freq[0], 0)), 0)
        hf[1] = torch.unsqueeze(self.hf2_conv_block_1(torch.unsqueeze(high_freq[1], 0)), 0)
        hf[2] = torch.unsqueeze(self.hf3_conv_block_1(torch.unsqueeze(high_freq[2], 0)), 0)

        return self.op([lf_conv_block_2, hf])

