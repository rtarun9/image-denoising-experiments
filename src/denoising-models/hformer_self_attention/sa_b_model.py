import torch
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

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

from pytorch_wavelets import DWT2D, IDWT2D

# Get the approximation, coarse high freq and fine + mid high freq.
def get_approximation_and_coarse_highpass_wavelet(image):
    xfm = DWT2D(J=3, mode='symmetric', wave='db3').cuda()
    yl, yh = xfm(image)
    
    coarse_highpass = yh[2]
    
    horizontal = coarse_highpass[:,:,0,:,:]
    vertical = coarse_highpass[:,:,1,:,:]
    diagonal = coarse_highpass[:,:,2,:,:]
    
    return yl, horizontal + vertical + diagonal, yh[0] , yh[1]

class DepthWiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                               groups=in_channels, bias=bias, padding='same').cuda()
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, bias=bias).cuda()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ConvBlock(torch.nn.Module):
    def __init__(self, num_channels, width, height, device='cuda'):
        super(ConvBlock, self).__init__()
        
        self.zero_padding = torch.nn.ZeroPad2d(padding=3).to(device)

        self.dwconv_approx = DepthWiseSeparableConv2d(num_channels, num_channels, 7).to(device)
        self.dwconv_high_freq = DepthWiseSeparableConv2d(num_channels, num_channels, 7).to(device)


        self.linear_1_approx = torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=7).to(device)
        self.linear_1_high_freq = torch.nn.Conv2d(num_channels, num_channels * 4, kernel_size=7).to(device)

        self.gelu_approx = torch.nn.GELU().to(device)
        self.gelu_high_freq = torch.nn.GELU().to(device)

        self.linear_2_approx = torch.nn.Conv2d(num_channels * 4, num_channels, kernel_size=1).to(device)
        self.linear_2_high_freq = torch.nn.Conv2d(num_channels * 4, num_channels, kernel_size=1).to(device)
        
        self.idwt = IDWT2D(wave='db3').to(device)
        
    def forward(self, images):
        residual = images
        approx, coarse_high_freq, fine_high_freq, mid_high_freq = get_approximation_and_coarse_highpass_wavelet(images)
       
        # Path a > Approximation (low frequency component pass) along with the fine and coarse details of high frequency components.
        # Now, why is fine and coarse not touched at all in the convolution? Because some important components might be present and we want to preserve them. 
        approx_images = self.zero_padding(approx)
        approx_images = self.dwconv_approx(approx_images)
        approx_images = self.linear_1_approx(approx_images)
        approx_images = self.gelu_approx(approx_images)
        approx_images = self.linear_2_approx(approx_images)

        path_a_output = approx_images + approx 

        # Path b > Coarse high frequency component pass.
        high_freq_components = self.zero_padding(coarse_high_freq)
        high_freq_components = self.dwconv_high_freq(high_freq_components)
        high_freq_components = self.linear_1_high_freq(high_freq_components)
        high_freq_components = self.gelu_high_freq(high_freq_components)
        high_freq_components = self.linear_2_high_freq(high_freq_components)

        path_b_output = high_freq_components + coarse_high_freq 
        path_b_output = torch.stack([path_b_output, path_b_output, path_b_output], 2)
           
        high_freq = [None]  * 3
        high_freq[0] = fine_high_freq
        high_freq[1] = mid_high_freq
        high_freq[2] = path_b_output 

        # Perform IDWT.
        result =  self.idwt((path_a_output, high_freq))

        return result + residual
 
import torch
import torch.nn as nn
import torch.nn.functional as F

class HformerBlock(nn.Module):
    def __init__(self, maxpool_kernel_size, attention_scaling_factor, num_channels, width, height):
        super(HformerBlock, self).__init__()

        self.k = maxpool_kernel_size
        self.dk = attention_scaling_factor
        self.c = num_channels

        # Layers for path A (i.e DWConv, Norm, Linear, Gelu, Linear)
        # Depth-wise conv (applying a separate filter for each image channel)
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

        # Defining the layers for the stage where the input is the output of path A and path B
        self.norm = nn.LayerNorm([num_channels, height, width], elementwise_affine=False)  
        self.linear_1 = nn.Conv2d(num_channels, num_channels * 4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Conv2d(num_channels * 4, num_channels, kernel_size=1)

    def forward(self, images):
        # In the Hformer block, the input takes 2 paths.
        # In path A, images undergo depth-wise convolution (DWConv based perceptual module)
        # In path B, images undergo a transformer module with lightweight self-attention module (self-attention after max pooling)

        # Code for path A.
        path_a_output = self.depthwise_conv(images)

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
        path_b_output = attention.view(-1, path_a_output.size(1), path_a_output.size(2), path_a_output.size(3))

        combined_path_output = path_a_output + path_b_output


        x = combined_path_output
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)


        return x

   
class SABModel(torch.nn.Module):
    def __init__(self, num_channels,width=64, height=64, device='cuda'):
        super(SABModel, self).__init__()
        
        self.ip = InputProjectionLayer(num_channels=num_channels).to(device)
        self.op = OutputProjectionLayer(num_channels=num_channels).to(device)
    
        self.conv_block_1 = ConvBlock(num_channels=num_channels, width=width, height=height).to(device)
        self.conv_block_2 = ConvBlock(num_channels=num_channels, width=width, height=height).to(device) 
        
        self.downsampling_layer_1 = torch.nn.Conv2d(num_channels, num_channels * 2, kernel_size=(2,2), stride=2).to(device)
        
        self.upsampling_layer_1 = torch.nn.ConvTranspose2d(num_channels * 2, num_channels, kernel_size=(2, 2), stride=2).to(device)
         
        self.hformer_block_1 = HformerBlock(maxpool_kernel_size=(2,2), attention_scaling_factor=1, num_channels=num_channels * 2, width=width // 2, height=height // 2).to(device)

        
    def forward(self, images):
        image_after_ip = self.ip(images)
        conv_block_1_output = self.conv_block_1(image_after_ip)
        downsampling_1_output = self.downsampling_layer_1(conv_block_1_output)
        hformer_block_1_output = self.hformer_block_1(downsampling_1_output)
        upsampling_2_output = self.upsampling_layer_1(hformer_block_1_output)
        conv_block_2_output = self.conv_block_2(upsampling_2_output) + conv_block_1_output
        
        output = self.op(conv_block_2_output) + images
        
        return output 
 