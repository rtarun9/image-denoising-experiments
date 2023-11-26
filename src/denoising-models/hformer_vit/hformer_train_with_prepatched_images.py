import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from tf_data_importer import load_training_tf_dataset

from hformer_model import get_hformer_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def split_dataset(dataset, split_ratio=0.8):
    total_samples = len(dataset)
    train_size = int(total_samples * split_ratio)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset

def train_model(training_dataset, epochs, trained_model_file_name, history_file_name):        
    # Testing if model can be compiled
    # From the paper,
    # The batch size is 16 through 4000 epochs. 
    # The ADAM-W optimizer was used to minimize the mean squared error loss, and the learning rate was 1.0 × 10−5
    # AdamW cannot be used with tf2.10, so revering to Adam.

    model = get_hformer_model(num_channels_to_be_generated=1, name="hformer_model_64_channel")

    # Train validation split.
    train_dataset, val_dataset = split_dataset(training_dataset, split_ratio=0.8)

    model.build(input_shape=(None, 64, 64, 1)) 
    model.compile(tf.keras.optimizers.Adam(learning_rate=1.0 * 10**-5), metrics='accuracy', loss='mse')
    
    history = model.fit(train_dataset, epochs=epochs,  validation_data=val_dataset)
    
    # Save the model weights to an HDF5 file
    # We cant use model.save as subclassing API is used here.
    model.save_weights(trained_model_file_name)
    
    # Save the training history
    np.save(history_file_name, history.history)
    
def main():
    training_dataset = load_training_tf_dataset(low_dose_ct_training_dataset_dir='../../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_as_patches=True, load_limited_images=True, num_images_to_load=2)
        
    trained_model_file_name = 'hformer_1_epoch.h5'
    history_file_name = 'hformer_1_epoch_history.npy'
    
    print('training dataset' , training_dataset)
    
    train_model(training_dataset, 1, trained_model_file_name, history_file_name)
    
    print('model trained successfully with name : ', trained_model_file_name)
    print('saved history in file with name : ', history_file_name)
    
if __name__ ==  "__main__":
    main()