from logging import PercentStyle
import sys

import tensorflow as tf
import numpy as np
import pywt

from tf_data_importer import load_training_tf_dataset

from hformer_model_extended import get_hformer_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import backend as K

from tensorflow import keras
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DMWT as DMWT


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def split_dataset(dataset, split_ratio=0.8):
    total_samples = len(dataset)
    train_size = int(total_samples * split_ratio)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset

# Define the custom PSNR metric
def psnr(y_true, y_pred):
    # Ensure the images have the same number of channels
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Calculate the MSE
    mse = K.mean(K.square(y_true - y_pred))

    # Calculate the PSNR
    max_pixel = 1.0  # Assuming pixel values are normalized between 0 and 1
    psnr_value = 10.0 * K.log((max_pixel ** 2) / mse) / tf.math.log(10.0)

    return psnr_value 

def min_max_normalize(img):
    min_val = tf.reduce_min(img)
    max_val = tf.reduce_max(img)
    normalized_img = (img - min_val) / (max_val - min_val)
    return normalized_img

def dwt_high(x):
    x1 = x[:, 0::2, 0::2, :]  # x(2i−1, 2j−1)
    x2 = x[:, 1::2, 0::2, :]  # x(2i, 2j-1)
    x3 = x[:, 0::2, 1::2, :]  # x(2i−1, 2j)
    x4 = x[:, 1::2, 1::2, :]  # x(2i, 2j)
    
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4
    
    return x_LH + x_HL + x_HH

def perceptual_loss_high(y_true, y_pred):
    dwt_true = dwt_high(y_true)
    dwt_pred = dwt_high(y_pred)

    # Calculate MSE on the high-frequency component (x_LH, x_HL, x_HH)
    mse_low_freq = tf.losses.mean_squared_error(dwt_true, dwt_pred)

    return mse_low_freq

def dwt_low(x):
    x1 = x[:, 0::2, 0::2, :]  # x(2i−1, 2j−1)
    x2 = x[:, 1::2, 0::2, :]  # x(2i, 2j-1)
    x3 = x[:, 0::2, 1::2, :]  # x(2i−1, 2j)
    x4 = x[:, 1::2, 1::2, :]  # x(2i, 2j)
    x_LL = x1 + x2 + x3 + x4
    return x_LL

def perceptual_loss_low(y_true, y_pred):
    dwt_true = dwt_low(y_true)
    dwt_pred = dwt_low(y_pred)

    # Calculate MSE on the low-frequency component (x_LL)
    mse_low_freq = tf.losses.mean_squared_error(dwt_true, dwt_pred)

    return mse_low_freq

def perceptual_wavelet_loss(y_true, y_pred):
    y_true, y_pred = min_max_normalize(y_true), min_max_normalize(y_pred)
    
    loss_low = perceptual_loss_low(y_true, y_pred)
    loss_high = perceptual_loss_high(y_true, y_pred)
    
    low_weight = 0.3
    high_weight = 0.7
    
    return loss_low * low_weight + loss_high * high_weight
    
perceptual_loss_high.__name__ = "perceptual_loss_high"
perceptual_loss_low.__name__ = "perceptual_loss_low"


def train_model(training_dataset, epochs, trained_model_file_name, history_file_name):        
    # Testing if model can be compiled
    # From the paper,
    # The batch size is 16 through 4000 epochs. 
    # The ADAM-W optimizer was used to minimize the mean squared error loss, and the learning rate was 1.0 × 10−5
    # AdamW cannot be used with tf2.10, so revering to Adam.

    model = get_hformer_model(num_channels_to_be_generated=64, name="hformer_model_64_channel")

    # Train validation split.
    train_dataset, val_dataset = split_dataset(training_dataset, split_ratio=0.8)

    model.build(input_shape=(100, 64, 64, 1)) 

    model.compile(tf.keras.optimizers.Adam(learning_rate=1.0 * 10**-5), metrics=[psnr, 'accuracy'], loss=perceptual_wavelet_loss, run_eagerly=True)
    print(model.summary())
    
    checkpoint_filepath = "weights_sa/hformer_sa_epochs_{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True)    

    history = model.fit(train_dataset, epochs=epochs,  verbose=1, validation_data=val_dataset, batch_size=2, callbacks=[checkpoint])
    
    # Save the model weights to an HDF5 file
    # We cant use model.save as subclassing API is used here.
    model.save_weights(trained_model_file_name)
    
    # Save the training history
    np.save(history_file_name, history.history)
    
def main():
    training_dataset = load_training_tf_dataset(low_dose_ct_training_dataset_dir='../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_as_patches=True, load_limited_images=True, num_images_to_load=10000)

    trained_model_file_name = 'weights_sa/hformer_wavelet.h5'
    history_file_name = 'weights_sa/hformer_wavelet.npy'
    
    print('training dataset' , training_dataset)
    
    train_model(training_dataset, 50, trained_model_file_name, history_file_name)
    
    print('model trained successfully with name : ', trained_model_file_name)
    print('saved history in file with name : ', history_file_name)
    
if __name__ ==  "__main__":
    main()
