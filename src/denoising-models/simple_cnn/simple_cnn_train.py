# The objective is to train a simple CNN model using tensorflow and save the trained model as a .h5 file.
# The loss function used here is PSNR.

import sys
sys.path.append('../')  

from data_importer import load_training_images
from data_importer import load_testing_images

from custom_loss_functions import psnr

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras import backend as K

from sklearn.model_selection import train_test_split

# Function to build a simple CNN model
def build_simple_cnn_model(input_shape):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(3, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(512 * 512 * 1, activation='linear'))
    model.add(layers.Reshape((512, 512, 1)))

    return model

def train_model(x_train, x_val, y_train, y_val, epochs, trained_model_file_name, history_file_name):
    input_shape = x_train[0].shape
    simple_cnn_model = build_simple_cnn_model(input_shape)

    # Compile the model
    simple_cnn_model.compile(optimizer='adam', loss=psnr, metrics=['accuracy'])

    history = simple_cnn_model.fit(x_train, y_train, epochs=epochs, batch_size=4, validation_data=(x_val, y_val))
    
    # Save the model to an HDF5 file
    simple_cnn_model.save(trained_model_file_name)
    
    # Save the training history
    np.save(history_file_name, history.history)

def main():
    x_data, y_data = load_training_images(low_dose_ct_training_dataset_dir='../../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_limited_images=True, num_images_to_load=10)
    
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    
    trained_model_file_name = 'simple_cnn_10_epoch.h5'
    history_file_name = 'simple_cnn_10_epoch_history.npy'
    
    train_model(x_train, x_val, y_train, y_val, 10, trained_model_file_name, history_file_name)
    
    print('model trained successfully with name : ', trained_model_file_name)
    print('saved history in file with name : ', history_file_name)
    
if __name__ ==  "__main__":
    main()