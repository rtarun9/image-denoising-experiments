import torch
import numpy as np

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
        self.layer_norm = torch.nn.LayerNorm([64, num_channels, width + 6, height + 6], elementwise_affine=False)
        self.linear_1 = torch.nn.Conv2d(num_channels, num_channels , kernel_size=7)
        self.gelu = torch.nn.GELU()
        self.linear_2 = torch.nn.Conv2d(num_channels , num_channels, kernel_size=1) 
        
    def forward(self, images):
        residual = images
        
        images = self.zero_padding(images)
        images = self.dwconv(images)
        images = self.layer_norm(images)
        images = self.linear_1(images)
        images = self.gelu(images)
        images = self.linear_2(images)
        return images + residual 
    
class HformerBlock(torch.nn.Module):
    def __init__(self, maxpool_kernel_size, attention_scaling_factor, num_channels, width, height):
        super(HformerBlock, self).__init__()

        self.k = maxpool_kernel_size
        self.dk = attention_scaling_factor
        self.c = num_channels

        self.depthwise_conv = DepthWiseSeparableConv2d(num_channels, num_channels, 7) 

        self.maxpool = torch.nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size)

        self.k_linear = torch.nn.Linear(self.c, self.c, bias=False)
        self.q_linear = torch.nn.Linear(self.c, self.c, bias=False)
        self.v_linear = torch.nn.Linear(self.c, self.c, bias=False)
        self.self_attention_linear = torch.nn.Conv2d(num_channels, num_channels, kernel_size=1)

        self.norm = torch.nn.LayerNorm([num_channels, height, width], elementwise_affine=False)  
        self.linear_1 = torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=1)
        self.gelu = torch.nn.GELU()
        self.linear_2 = torch.nn.Conv2d(num_channels * 4, num_channels, kernel_size=1)

    def forward(self, images):
        path_a_output = self.depthwise_conv(images)

        max_pooled_images = self.maxpool(images)

        flattened_max_pooled_images = max_pooled_images.view(max_pooled_images.size(0), -1, self.c)

        q = self.q_linear(flattened_max_pooled_images)
        q = q.permute(0, 2, 1)

        k = self.k_linear(flattened_max_pooled_images)

        flattened_images = images.view(images.size(0), -1, self.c)

        v = self.v_linear(flattened_images)

        attention = torch.matmul(q, k)
        attention = torch.nn.functional.softmax(attention / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32)), dim=-1)

        attention = torch.matmul(v, attention)

        path_b_output = attention.view(-1, path_a_output.size(1), path_a_output.size(2), path_a_output.size(3))

        combined_path_output = path_a_output + path_b_output

        x = combined_path_output
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)

        return x
    
class HformerModel(torch.nn.Module):
    def __init__(self, num_channels,width=64, height=64):
        super(HformerModel, self).__init__()
        
        self.ip = InputProjectionLayer(num_channels=num_channels)
        self.op = OutputProjectionLayer(num_channels=num_channels)
    
        self.conv_block_1 = ConvBlock(num_channels=num_channels, width=width, height=height)
        self.conv_block_2 = ConvBlock(num_channels=num_channels, width=width, height=height) 
        
        self.downsampling_layer_1 = torch.nn.Conv2d(num_channels, num_channels * 2, kernel_size=(2,2), stride=2)
        self.downsampling_layer_2 = torch.nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=(2,2), stride=2)
        
        self.upsampling_layer_1 = torch.nn.ConvTranspose2d(num_channels * 4, num_channels * 2, kernel_size=(2, 2), stride=2)
        self.upsampling_layer_2 = torch.nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=(2, 2), stride=2)
         
        self.hformer_block_1 = HformerBlock(maxpool_kernel_size=(2,2), attention_scaling_factor=1, num_channels=num_channels * 2, width=width // 2, height=height // 2)
        self.hformer_block_2 = HformerBlock(maxpool_kernel_size=(2,2), attention_scaling_factor=1, num_channels=num_channels * 4, width=width // 4, height=height // 4)
        self.hformer_block_3 = HformerBlock(maxpool_kernel_size=(2,2), attention_scaling_factor=1, num_channels=num_channels * 2, width=width // 2, height=height // 2)
        
    def forward(self, images):
        image_after_ip = self.ip(images)
        conv_block_1_output = self.conv_block_1(image_after_ip)
        downsampling_1_output = self.downsampling_layer_1(conv_block_1_output)
        hformer_block_1_output = self.hformer_block_1(downsampling_1_output)
        downsampling_2_output = self.downsampling_layer_2(hformer_block_1_output)
        hformer_block_2_output = self.hformer_block_2(downsampling_2_output)
        upsampling_1_output = self.upsampling_layer_1(hformer_block_2_output)
        hformer_block_3_output = self.hformer_block_3(upsampling_1_output) + hformer_block_1_output
        upsampling_2_output = self.upsampling_layer_2(hformer_block_3_output)
        conv_block_2_output = self.conv_block_2(upsampling_2_output) + conv_block_1_output
        
        output = self.op(conv_block_2_output)
        
        return output + images