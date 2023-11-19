
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import backend as K

# Define the custom PSNR loss function
def psnr(y_true, y_pred):
    # Ensure the images have the same number of channels
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Calculate the MSE
    mse = K.mean(K.square(y_true - y_pred))

    # Calculate the PSNR
    max_pixel = 1.0  # Assuming pixel values are normalized between 0 and 1
    psnr_value = 10.0 * K.log((max_pixel ** 2) / mse) / tf.math.log(10.0)

    return -psnr_value  # Return the negative PSNR as a loss (to minimize)

