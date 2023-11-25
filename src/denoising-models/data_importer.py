import numpy as np
import pydicom
import os

# Takes in input as a .ima file (or any file readable by dicom) and returns a pixel array
def read_image(image_path):
    data = pydicom.dcmread(image_path)
    return data.pixel_array

# A function that returns training images from the LowDoseCT Challenge dataset (link : https://www.aapm.org/grandchallenge/lowdosect/)
# If load_limited_images is True, it will load number of images that are specified in images_to_load.
# Else, the entire dataset will be loaded.
def load_training_images(low_dose_ct_training_dataset_dir='../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_limited_images=False, num_images_to_load=10):
    
    training_filepaths_x = []   # i.e the QD (quarter dose) images (noisy images)
    training_filepaths_y = []   # i.e the FD (full dose) images (clean images)

    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
        
            if "QD" in file_name:
                training_filepaths_x.append(file_path)
            elif "FD" in file_name:
                training_filepaths_y.append(file_path)
            
    training_filepaths_x.sort()
    training_filepaths_y.sort()
    
    if load_limited_images:
        training_filepaths_x = training_filepaths_x[:num_images_to_load]
        training_filepaths_y = training_filepaths_y[:num_images_to_load]
        
    training_images_x = np.array([np.expand_dims(read_image(path), axis=-1) for path in training_filepaths_x])
    training_images_y = np.array([np.expand_dims(read_image(path), axis=-1) for path in training_filepaths_y])
        
    # Normalize pixel values to the range [0, 1]
    training_images_x = training_images_x / np.max(training_images_x)
    training_images_y = training_images_y / np.max(training_images_y)
    
    print('loaded training images x and y of len : ', len(training_images_x), len(training_images_y), ' respectively')
    print('type of train images x : ', training_images_x[0].dtype)
    print('range of values in train images : ', np.min(training_images_x[0]), np.max(training_images_x[0]))
    print('type of train images y : ', training_images_y[0].dtype)
    
    return training_images_x, training_images_y

# A function that returns testing images from the LowDoseCT Challenge dataset (link : https://www.aapm.org/grandchallenge/lowdosect/)
# If load_limited_images is True, it will load number of images that are specified in images_to_load.
# Else, the entire dataset will be loaded.
def load_testing_images(low_dose_ct_testing_dataset_dir='../../../Dataset/LowDoseCTGrandChallenge/Testing_Image_Data', load_limited_images=False, num_images_to_load=10):
    
    testing_filepaths_x = []   # i.e the QD (quarter dose) images (noisy images)

    for root, folder_name, file_names in os.walk(low_dose_ct_testing_dataset_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
        
            if "QD" in file_name:
                testing_filepaths_x.append(file_path)
            
    testing_filepaths_x.sort()
    
    if load_limited_images:
        testing_filepaths_x = testing_filepaths_x[:num_images_to_load]
        
    testing_images_x = np.array([np.expand_dims(read_image(path), axis=-1) for path in testing_filepaths_x])

    # Normalize pixel values to the range [0, 1]
    testing_images_x = testing_images_x / np.max(testing_images_x)
    
    print('loaded testing images x of len : ', len(testing_images_x))
    print('type of test images x : ', testing_images_x[0].dtype)
    print('range of values in test images : ', np.min(testing_images_x[0]), np.max(testing_images_x[0]))
    
    return testing_images_x

