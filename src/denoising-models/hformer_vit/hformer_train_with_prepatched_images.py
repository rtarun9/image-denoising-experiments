import einops

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks

from data_importer import load_training_images
from data_importer import load_testing_images

from sklearn.model_selection import train_test_split

# For sake of efficiency, tf.image has functions that will extract patches from images.
# Creating a custom layer for ease of use when creating the model.
# https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html

# Takes as input patch_size, which will be same along width and height
# As hformer uses overlapping slices to increase number of training samples,
# set the stride to a values less than patch_size
# The output shape is num_patches, patch_height, patch_width, channel
# The reason for not having batch_index in the output is that this model doesnt seem to bother merging the patches (i.e merging is done, but thats at the very end)
# Also, the conv block layer expects the shape of image to be num_batches, height, width, so doing this makes sense.
# For compuing which patches belong to which image, simple division can be done.
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def call(self, images):
        batch_size = tf.shape(images)[0]
        channels = images.shape[-1]
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [-1, self.patch_size, self.patch_size, channels])
        return patches
    
    # Custom tf layer for convolutional block.
# Official / Unofficial documentation for the constituent layers / blocks:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
# https://github.com/dongkyuk/ConvNext-tensorflow/blob/main/model/convnext.py
# https://keras.io/examples/vision/edsr/
# https://github.com/martinsbruveris/tensorflow-image-models/blob/d5f54175fb91e587eb4d2acf81bb0a7eb4424f4a/tfimm/architectures/convnext.py#L146

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ConvolutionBlock, self).__init__()
        self.layers = tf.keras.Sequential([
            # Layer 0 : Padding to ensure that the output size is same as input size.
            tf.keras.layers.ZeroPadding2D(padding=3),
            
            # Layer 1 : Depth wise convolution (7x7)
            tf.keras.layers.DepthwiseConv2D(7, depth_multiplier=1),
            
            # Layer 2 : Layer normalization
            tf.keras.layers.LayerNormalization(),
        
            # Layer 3 : Linear (1)
            tf.keras.layers.Conv2D(1, kernel_size=1),
            
            # Layer 4 : GELU activation
            tf.keras.layers.Activation('gelu'),
            
            # Layer 5 : Linear (2)
            tf.keras.layers.Conv2D(1, kernel_size=1),
        ])
        
    def call(self, images):
        # print('[Conv block] input : ', images.shape)
        # Here, assuming that images are of the shape (num_images, image_height, image_width, num_channels)
        output = self.layers(images)
        # print('[Conv block] output : ', output.shape)
        
        # Added the residual / skip connection
        residual = images
        
        # Apply a 1x1 convolution to match the output channel dimension if necessary
        if output.shape[-1] != residual.shape[-1]:
            residual = tf.keras.layers.Conv2D(output.shape[-1], kernel_size=1)(residual)

        output = output + residual

        return output        
    

# Note  : Max pooling kernel size is taken as value for stride in the max pooling layer.
# Attention scaling factor depends on the network depth. It is the (1 / sqrt(dk)) that is mentioned in the paper.
class HformerBlock(tf.keras.layers.Layer):
    def __init__(self, depth_wise_conv_kernel_size, maxpool_kernel_size, attention_scaling_factor, **kwargs):
        super(HformerBlock, self).__init__(**kwargs)

        # Saving the layer input parameters.
        self.depth_wise_conv_kernel_size = depth_wise_conv_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.dk = attention_scaling_factor

        # Defining the layers required by the HformerBlock.
        
        # Depth wise conv (i.e applying a separate filter for each image channel)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            (depth_wise_conv_kernel_size, depth_wise_conv_kernel_size),
            strides=(1, 1),
            padding='same',
            use_bias=False)
        
        # Max pooling layers (will be used in attention computation for Keys and Queries)
        self.maxpool = tf.keras.layers.MaxPooling2D((maxpool_kernel_size, maxpool_kernel_size), strides=(maxpool_kernel_size, maxpool_kernel_size))
                
        # These are used in the Conv block also.
        self.gelu = tf.keras.layers.Activation('gelu')
        self.norm = tf.keras.layers.LayerNormalization()
        self.linear = tf.keras.layers.Conv2D(1, kernel_size=1)

        # Dynamically set C based on input shape (Doesn't work, so just setting it to 1 for now)
        # Based on the paper it should be number of patches, but for temporary time we are splitting the image into patches for input itself.
        self.C = 1

        # Create variables based on dynamically determined C 
        # (Q, K, V) are used for attention.
        # This is clear from the paper which mentions :
        # The (original) self-attentive mechanism first generates the corresponding query e
        # (q), Key (K) and Value (V)
        # The output of K and Q linear are : HW/k^2 X C
        
        self.q_linear = tf.keras.layers.Dense(self.C, use_bias=False)
        self.k_linear = tf.keras.layers.Dense(self.C, use_bias=False)
        self.v_linear = tf.keras.layers.Dense(self.C, use_bias=False)

    def call(self, images):
        
        # In the Hformer block, the input takes 2 paths.
        # In path A, images undergoe depth wise convolution (DWConv based perceptual module)
        # In path B, images undergoe a transformer module with lightweight self attention module (i.e self attention after maxpooling)
        
        # Code for path A.
        path_a_output = self.depthwise_conv(images)
        
        # print('depth wise conv output shape : ', path_a_output.shape)
        
        # Code for path B.
        max_pooled_images = self.maxpool(images)
                    
        # print('shape after max pooling : ', max_pooled_images.shape)
        
        # Attention module computation.
        # Q, K, V all pass through linear layers. 
        # q and k are reshaped into HW / (max pool kernel size)^2 X C
        # print('input shape expected by q_linear, k_linear and v_linear : ', self.C)
        
        q = self.q_linear(max_pooled_images)
        # print('shape of q after applyling q_linear : ', q.shape)
        q = tf.reshape(q, (-1, self.C, max_pooled_images.shape[1] * max_pooled_images.shape[2]))
        # print('shape of q after reshaping  : ', q.shape)
        
        k = self.k_linear(max_pooled_images)
        k = tf.reshape(k, (-1, self.C, max_pooled_images.shape[1] *  max_pooled_images.shape[2]))

        # print('q and k shape : ', k.shape)
        
        v = self.v_linear(images)
        v = tf.reshape(v, (-1, self.C, images.shape[1] * images.shape[2]))
        
        # print('shape of v :', v.shape)

        # Computation of attention
        # attention = ((K'T Q) / sqrt(dk)) V
        attention = tf.matmul(q, k, transpose_b=True)
        # print('q * k shape : ', attention.shape)
        
        # As per paper, after performing softmax, we obtain a attention score matrix with dimension C x C.
        # The scaling factor mentioned in the paper (i.e 1/sqrt(dk)) is based on network depth.
        attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.dk, tf.float32)))

        # Now, the final attention map is obtained by multiplied v and attention.
        attention = tf.matmul(attention, v)
        
        # print ('final attention shape : ', attention.shape)
        
        # Now, attention is reshaped into the same dimensions as the input image.
        path_b_output = tf.reshape(attention, (-1, path_a_output.shape[1], path_a_output.shape[2], path_a_output.shape[3]))
        
        # print('path b output shape : ', path_b_output.shape)
        
        combined_path_output = path_b_output + path_a_output
        
        # print('combined path output shape : ', combined_path_output.shape)
            
        x = combined_path_output
        x = self.norm(x)
        x = self.linear(x)
        # print('shape after hformer block linear 1 : ', x.shape)
        x = self.gelu(x)
        # print('shape after hformer block gelu : ', x.shape)
        x = self.linear(x)
        
        return x

# Documentation of subclassing API : https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing

# Note : Assumes x and y data are already split into image patches (overlapping or non overlapping)
class Hformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # While this is here, the model takes as input the image patches.
        # A slight deviation to the original architecture
        self.patch_extraction = PatchExtractor(patch_size=32, stride=32)
        
        self.conv_net_block = ConvolutionBlock()
        
        # As per the paper, 2x2 conv layers are used for both upsampling and downsampling.
        self.down_sampling_layer = tf.keras.layers.Conv2D(1, (2,2), (2,2))
        
        # Conv2D transpose : Deconv layer
        self.up_sampling_layer = tf.keras.layers.Conv2DTranspose(1, (2, 2), (2, 2))
        
        self.hformer_block = HformerBlock(depth_wise_conv_kernel_size=7, maxpool_kernel_size=2, attention_scaling_factor=1)
        
    def call(self, images):
        input_image_shape = images.shape
        
        # print('shape of input images : ', images.shape)
        
        # Split image into patches
        # image_patches = self.patch_extraction(images)
        # The model assumes this has been done already.
        image_patches = images
        
        # print('size of image patches : ', image_patches.shape)
        
        # First conv block filtering
        conv_block_1_output = self.conv_net_block(image_patches)
        
        # print('shape after conv block : ', conv_block_1_output.shape)
        
        # Downsampling images from (C X H X W) to (2C X H/2 X W/2)
        x = self.down_sampling_layer(conv_block_1_output)
        
        # print('hformer input shape : ', x.shape)
        
        # Hformer block application
        hformer_block_1_output = self.hformer_block(x)
        
        # Downsampling imagesm from (2C X H/2 X W/2) to (4C X H/4 X W/4)
        x = self.down_sampling_layer(hformer_block_1_output)
        
        # print('shape after downsampling : ', x.shape)
        
        x = self.hformer_block(x)
        
        # Upsampling block 1
        x = self.up_sampling_layer(x)
        
        # Hformer block application and skip connection.
        x = self.hformer_block(x)
        
        x = x + hformer_block_1_output
        
        # Upsampling image to 2C X H/2 X W/2
        x = self.up_sampling_layer(x)
        
        # Conv block filtering + skip connection.
        x = self.conv_net_block(x)
        
        x = x + conv_block_1_output
        
        # print('hformer model output shape : ', x.shape)
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

def train_model(x_train, x_val, y_train, y_val, epochs, trained_model_file_name, history_file_name):        
    # Testing if model can be compiled
    # From the paper,
    # The batch size is 16 through 4000 epochs. 
    # The ADAM-W optimizer was used to minimize the mean squared error loss, and the learning rate was 1.0 × 10−5
    # AdamW cannot be used with tf2.10, so revering to Adam.
    patch_extractor = PatchExtractor(patch_size=32, stride=32)
    model = Hformer()

    model.build(patch_extractor(x_train).shape) 
    model.compile(tf.keras.optimizers.Adam(learning_rate=1.0 * 10**-5), metrics='accuracy', loss='mse')


    x_train_patches = patch_extractor(x_train)
    y_train_patches = patch_extractor(y_train)

    x_val_patches = patch_extractor(x_val)
    y_val_patches = patch_extractor(y_val)
    
    history = model.fit(x_train_patches, y_train_patches, epochs=epochs, batch_size=64, validation_data=(x_val_patches, y_val_patches))

    # Save the model weights to an HDF5 file
    # We cant use model.save as subclassing API is used here.
    model.save_weights('my_model_weights.h5')
    
    # Save the training history
    np.save(history_file_name, history.history)
    
def main():
    x_data, y_data = load_training_images(low_dose_ct_training_dataset_dir='../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_limited_images=True, num_images_to_load=200)
    
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    trained_model_file_name = 'hformer_10_epoch.h5'
    history_file_name = 'hformer_10_epoch_history.npy'
    
    train_model(x_train, x_val, y_train, y_val, 1, trained_model_file_name, history_file_name)
    
    print('model trained successfully with name : ', trained_model_file_name)
    print('saved history in file with name : ', history_file_name)
    
if __name__ ==  "__main__":
    main()