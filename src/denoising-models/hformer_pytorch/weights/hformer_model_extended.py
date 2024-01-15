import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from tf_data_importer import load_training_tf_dataset

from sklearn.model_selection import train_test_split

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow import keras
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DMWT as DMWT


# For sake of efficiency, tf.image has functions that will extract patches from images.
# Creating a custom layer for ease of use when creating the model.
# https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html


# Takes as input patch_size, which will be same along width and height
# As hformer uses overlapping slices to increase number of training samples, set the stride to a values less than patch_size.

# NOTE : The output shape is num_patches, patch_height, patch_width, patch_depth.
# The Fig1 of paper mentions input to input projection layer is 1 X H X W. Output of this layer is H X W X 1

class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride,name):
        super(PatchExtractor, self).__init__(name=name)
        self.patch_size = patch_size
        self.stride = stride

    def call(self, images):
        # batch_size : number of images, which is not used.
        batch_size = tf.shape(images)[0] 
        # Expected to always be 1.
        patch_depth = images.shape[-1]
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, self.patch_size, self.patch_size, patch_depth])
        return patches
    
    

# NOTE : The paper does not mention what constitutes the input projection block.
# The ONLY thing mentioned in input is each image patch (that is 64x64). Input to the input projection block is 1 X H X W, and output is C X H X W

# NOTE (ASSUMPTIONS) : The input 'projection' block uses convolution (2x2) and C is the number of kernels.
# I chose 3x3 since the paper does not say what H, W.

# num_output_channels = C in the block diagram. NO activation function is used (assumed linear).
class InputProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, num_output_channels, kernel_size=(2, 2), name="input_projection_layer", **kwargs):
        super(InputProjectionLayer, self).__init__(name=name, **kwargs)

        # Define the convolutional layer
        self.convolution = tf.keras.layers.Conv2D(filters=num_output_channels,
                                                  kernel_size=kernel_size,
                                                  padding='same', name="convolution_layer")

    def call(self, inputs):
        output = self.convolution(inputs)
        return output
    
    

# NOTE : The paper does not mention what constitutes the output projection block.
# The ONLY thing mentioned in input is of shape C X H X W and output is 1 X H X W

# NOTE (ASSUMPTIONS) : The output 'projection' block uses transposed convolution 2x2) and C is the value of C (num_features_maps).

# NO activation function is used (assumed linear).
# kernel_size must MATCH whatever was given to InputProjectionLayer
class OutputProjectionLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_size=(2, 2), name="output_projection_layer", **kwargs):
        super(OutputProjectionLayer, self).__init__(name=name, **kwargs)

        # Define the deconvolutional layer
        self.transpose_conv2d = tf.keras.layers.Conv2DTranspose(filters=1,
                                                             kernel_size=kernel_size,
                                                             padding='same', name="transpose_convolution_2d_layer")

    def call(self, inputs):
        output = self.transpose_conv2d(inputs)
        return output



# Custom tf layer for convolutional block.

# Official / Unofficial documentation for the constituent layers / blocks:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
# https://github.com/dongkyuk/ConvNext-tensorflow/blob/main/model/convnext.py
# https://keras.io/examples/vision/edsr/
# https://github.com/martinsbruveris/tensorflow-image-models/blob/d5f54175fb91e587eb4d2acf81bb0a7eb4424f4a/tfimm/architectures/convnext.py#L146

# io_num_channels : Number of channels per image / patch for input and output.
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, io_num_channels, name):
        super(ConvolutionBlock, self).__init__(name=name)
        self.layers = tf.keras.Sequential([
            # Layer 0 : Padding to ensure that the output size is same as input size.
            # Padding is done by 3 because the DWConv will always be 7x7
            tf.keras.layers.ZeroPadding2D(padding=3, name="zero_padding_2d"),
            
            # Layer 1 : Depth wise convolution (7x7)
            tf.keras.layers.DepthwiseConv2D(7, depth_multiplier=1, name="depthwise_conv_2d"),
            
            # Layer 2 : Layer normalization
            tf.keras.layers.LayerNormalization(name="layer_normalization"),
        
            # Layer 3 : Linear (1)
            tf.keras.layers.Conv2D(io_num_channels * 4, kernel_size=1, name="linear_1x1_conv_2d_1"),
            
            # Layer 4 : GELU activation
            tf.keras.layers.Activation('gelu', name="gelu_activation"),
            
            # Layer 5 : Linear (2)
            tf.keras.layers.Conv2D(io_num_channels, kernel_size=1, name="linear_1x1_conv_2d_2"),
        ])
        
    def call(self, images):
        residual = images
                
        # Here, assuming that images are of the shape (num_images, image_height, image_width, num_channels)
        output = self.layers(images) + residual
        
        return output        
    
    # Note  : Max pooling kernel size is taken as value for stride in the max pooling layer.
# This is because we want NO overlaps between the 'pools' used in maxpooling.

# Implementation of a DWT / IDWT layer that can work when number of channels is > 1.

class CustomDWT(tf.keras.layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)
        
        self.dwt = DWT.DWT(name="db4", concat=0)
        
    def call(self, images):
        outputs = []
        for i in range(images.shape[-1]):
            outputs.append(self.dwt(tf.expand_dims(images[:,:,:,i], axis=-1)))
        return tf.convert_to_tensor(outputs)


        
class CustomIDWT(tf.keras.layers.Layer):
    def __init__(self, name):
        super().__init__(name=name)
        
        self.idwt = DWT.IDWT(concat=0)
        
    def call(self, images):
        outputs = []
        for i in range(images.shape[0]):
            outputs.append(self.idwt(images[i]))
        outputs = tf.convert_to_tensor(outputs) 
        outputs = tf.squeeze(outputs, axis=-1)
        outputs = tf.transpose(outputs, perm=[1, 2, 3, 0]) 

        return outputs

class HformerBlockWaveletInfused(tf.keras.layers.Layer):
    def __init__(self, maxpool_kernel_size, attention_scaling_factor, num_channels, name, **kwargs):
        super(HformerBlockWaveletInfused, self).__init__(name=name, **kwargs)

        # Saving the layer input parameters.
        self.k = maxpool_kernel_size
        self.dk = attention_scaling_factor
        self.c = num_channels

        # Defining the layers required by the HformerBlock.
        
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            (7, 7),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            name="depth_wise_conv_2d")

        self.maxpool = tf.keras.layers.MaxPooling2D((maxpool_kernel_size, maxpool_kernel_size),
                                                    strides=(maxpool_kernel_size, maxpool_kernel_size),
                                                    name="max_pooling_layer")

        self.v_linear = tf.keras.layers.Dense(self.c, use_bias=False, name="v_linear")
        
        self.k_linear = []
        self.q_linear = [] 
    
        for i in range(4):
            self.k_linear.append(tf.keras.layers.Dense(self.c, use_bias=False, name=f"k_linear_{i}"))
            self.q_linear.append(tf.keras.layers.Dense(self.c, use_bias=False, name=f"q_linear_{i}"))

        self.self_attention_linear = tf.keras.layers.Conv2D(num_channels, kernel_size=1, name="self_attention_linear")
        
        self.dwt = CustomDWT(name="custom_dwt")
        self.idwt = CustomIDWT(name="custom_idwt")
        
        self.norm = tf.keras.layers.LayerNormalization(name="layer_normalization")        
        self.linear_1 = tf.keras.layers.Conv2D(num_channels * 4, kernel_size=1, name="linear_1x1_conv_2d_1")
        self.gelu = tf.keras.layers.Activation('gelu', name="gelu_activation")
        self.linear_2 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, name="linear_1x1_conv_2d_2")

    def call(self, images):
        path_a_output = self.depthwise_conv(images)
        max_pooled_images = self.maxpool(images)

        # Attention module computation.
        wavelet_bands = self.dwt(max_pooled_images)
        wavelet_band_attentions = []
        
        flattened_images = tf.reshape(max_pooled_images, (tf.shape(max_pooled_images)[0], 
                                                          max_pooled_images.shape[1] * max_pooled_images.shape[2], 
                                                          self.c))
        v = self.v_linear(flattened_images)
        for i in range(4):
            wavelet_band = wavelet_bands[:,:,:,:,i]
            transformed_wavelet_band = tf.transpose(wavelet_band, perm=[1, 2, 3, 0])
            
            flattened_wavelet_band = tf.reshape(transformed_wavelet_band, 
                                                     (transformed_wavelet_band.shape[0], 
                                                      transformed_wavelet_band.shape[1] * transformed_wavelet_band.shape[2], 
                                                      self.c))
            q = self.q_linear[i](flattened_wavelet_band)
            q = tf.transpose(q, perm=[0, 2, 1])
            k = self.k_linear[i](flattened_wavelet_band)
            attention = tf.matmul(q, k)
            attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.dk, tf.float32)))
            attention = tf.matmul(v, attention)
            wavelet_band_attentions.append(tf.reshape(attention, (-1, 
                                                                  max_pooled_images.shape[1], 
                                                                  max_pooled_images.shape[2], 
                                                                  max_pooled_images.shape[3])))
        wavelet_band_attentions = tf.convert_to_tensor(wavelet_band_attentions)
        wavelet_band_attentions = tf.transpose(wavelet_band_attentions, perm=[4, 1, 2, 3, 0])

        # Replace the loop with a direct call to IDWT
        attention = self.idwt(wavelet_band_attentions)
        # Dynamically compute the target shape based on the dimensions of path_a_output
        target_shape = tf.concat([tf.shape(path_a_output)[:3], [self.c]], axis=0)
        path_b_output = tf.reshape(attention, target_shape)
        combined_path_output = path_a_output + path_b_output 
        x = combined_path_output
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        
        return x

# Documentation of subclassing API : https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing

# Note : Assumes x and y data are already split into image patches (overlapping or non overlapping)
class HformerModel(tf.keras.Model):
    def __init__(self, num_channels_to_be_generated, name):
        super().__init__(name=name)
        
        self.input_projection = InputProjectionLayer(num_output_channels=num_channels_to_be_generated, kernel_size=(2,2), name="input_projection_layer")
        self.output_projection = OutputProjectionLayer(kernel_size=(2,2), name="output_projection_layer")
        
        self.conv_net_block_1 = ConvolutionBlock(io_num_channels=num_channels_to_be_generated, name="conv_net_block_1")
        self.conv_net_block_2 = ConvolutionBlock(io_num_channels=num_channels_to_be_generated, name="conv_net_block_2")
        
        # As per the paper, 2x2 conv layers are used for both upsampling and downsampling.
        self.down_sampling_layer_1 = tf.keras.layers.Conv2D(num_channels_to_be_generated * 2, (2,2), (2,2), name="downsampling_layer_1")
        self.down_sampling_layer_2 = tf.keras.layers.Conv2D(num_channels_to_be_generated * 4, (2,2), (2,2), name="downsampling_layer_2")
        
        # Conv2D transpose : Deconv layer
        self.up_sampling_layer_1 = tf.keras.layers.Conv2DTranspose(num_channels_to_be_generated * 2, (2, 2), (2, 2), name="upsampling_layer_1")
        self.up_sampling_layer_2 = tf.keras.layers.Conv2DTranspose(num_channels_to_be_generated, (2, 2), (2, 2), name="upsampling_layer_2")
        
        self.hformer_block_1 = HformerBlockWaveletInfused(maxpool_kernel_size=2, attention_scaling_factor=1, num_channels=num_channels_to_be_generated * 2, name="hformer_block_1")
        self.hformer_block_2 = HformerBlockWaveletInfused(maxpool_kernel_size=2, attention_scaling_factor=1, num_channels=num_channels_to_be_generated * 4, name="hformer_block_2")
        self.hformer_block_3 = HformerBlockWaveletInfused(maxpool_kernel_size=2, attention_scaling_factor=1, num_channels=num_channels_to_be_generated * 2, name="hformer_block_3")
    
    def call(self, images):
        
        # Split image into patches
        # image_patches = self.patch_extraction(images)
        # The model assumes this has been done already.
        image_patches = images
        
        x = self.input_projection(image_patches)
        
        # First conv block filtering
        conv_block_1_output = self.conv_net_block_1(x)
        
        # Downsampling images from (C X H X W) to (2C X H/2 X W/2)
        x = self.down_sampling_layer_1(conv_block_1_output)
        
        # Hformer block application
        hformer_block_1_output = self.hformer_block_1(x)

        # Downsampling imagesm from (2C X H/2 X W/2) to (4C X H/4 X W/4)
        x = self.down_sampling_layer_2(hformer_block_1_output)
        
        x = self.hformer_block_2(x)
        
        # Upsampling block 1
        x = self.up_sampling_layer_1(x)
        
        # Hformer block application and skip connection.
        x = self.hformer_block_3(x) + hformer_block_1_output
        
        # Upsampling image to 2C X H/2 X W/2
        x = self.up_sampling_layer_2(x)
        
        
        # Conv block filtering + skip connection.
        x = self.conv_net_block_2(x) + conv_block_1_output
        
        x = self.output_projection(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "C": self.C
        })
        return config

    @classmethod
    def from_config(cls, config):
        layer = cls(**config)
        layer.build(input_shape=config["input_shape"])
        return layer
    
def get_hformer_model(num_channels_to_be_generated, name):
    return HformerModel(num_channels_to_be_generated=num_channels_to_be_generated, name=name)