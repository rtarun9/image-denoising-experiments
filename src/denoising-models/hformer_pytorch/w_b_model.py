
import torch
import os
import sys
from transformers import ConvNextV2Config, ConvNextV2Model
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



import torch
import torch.nn as nn
import torch.nn.functional as F

class WBlock(nn.Module):
    def __init__(self, maxpool_kernel_size, attention_scaling_factor, num_channels, width, height):
        super(WBlock, self).__init__()

        self.k = maxpool_kernel_size
        self.dk = attention_scaling_factor
        self.c = num_channels

        self.depthwise_conv = DepthWiseSeparableConv2d(num_channels, num_channels, 7) 

        # Defining the layers for path B (i.e max pooling, self-attention, linear)

        # Max pooling layers (used in attention computation for Keys and Queries)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size)

        # (Q, K, V) are used for attention.
        # This is clear from the paper which mentions:
        # The (original) self-attentive mechanism first generates the corresponding query e
        # (q), Key (K) and Value (V)
        # The output of K and Q linear are: HW/k^2 x C
        # Note that the vectors (output of self.k/q/v_linear) are PER channel (C).
        self.k_linear = nn.Linear(self.c, self.c, bias=False)
        self.q_linear = nn.Linear(self.c, self.c, bias=False)
        self.v_linear = nn.Linear(self.c, self.c, bias=False)
        self.self_attention_linear = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, images):
        # Code for path B.
        max_pooled_images = self.maxpool(images)

        # Attention module computation.
        # Q, K, V all pass through linear layers.
        # q and k are reshaped into HW / (max pool kernel size)^2 x C

        # The shape of max-pooled images is: (num_images, height', width', channels).
        # We are converting that into (num_images, height' * width', channel)
        flattened_max_pooled_images = max_pooled_images.view(max_pooled_images.size(0), -1, self.c)

        q = self.q_linear(flattened_max_pooled_images)
        # For the computation of the attention map, we need the shape of q to be num_images, num_channels, HW.
        # But now, it is num_images, HW, num_channels. The last 2 dimensions must be reversed.
        q = q.permute(0, 2, 1)

        k = self.k_linear(flattened_max_pooled_images)

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
        output = attention.view(-1, images.size(1), images.size(2), images.size(3))

        return output
        

import torch

class WModel(torch.nn.Module):
    def __init__(self, num_channels,width=64, height=64):
        super(WModel, self).__init__()
        
        self.ip = InputProjectionLayer(num_channels=num_channels)
        self.op = OutputProjectionLayer(num_channels=num_channels)
    
        self.unet_1= torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=num_channels, out_channels=num_channels, init_features=num_channels, pretrained=False)
        self.unet_2= torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=num_channels, out_channels=num_channels, init_features=num_channels, pretrained=False)

        self.downsampling_layer_1 = torch.nn.Conv2d(num_channels, num_channels * 2, kernel_size=(2,2), stride=2)
        
        self.upsampling_layer_1 = torch.nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=(2, 2), stride=2)
         
        self.hformer_block_1 = WBlock(maxpool_kernel_size=(2,2), attention_scaling_factor=1, num_channels=num_channels * 2, width=width // 2, height=height // 2)
        
    def forward(self, images):
        image_after_ip = self.ip(images)
        unet_block_1_output = self.unet_1(image_after_ip)
        downsampling_1_output = self.downsampling_layer_1(unet_block_1_output)
        hformer_block_1_output = self.hformer_block_1(downsampling_1_output)
        upsampling_2_output = self.upsampling_layer_1(hformer_block_1_output)
        unet_block_2= self.unet_2(upsampling_2_output) +unet_block_1_output 
        
        output = self.op(unet_block_2) + images
        
        return output 