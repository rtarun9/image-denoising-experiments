import numpy as np
import pydicom
import os

def load_scan(path):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    slices = [pydicom.read_file(path)]
    return slices


def get_pixels_hu(slices):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.int16(image)

def read_image(image_path):
    full_pixels =get_pixels_hu(load_scan(image_path))
    
    MIN_B= -1024.0
    MAX_B= 3072.0
    data = (full_pixels - MIN_B) / (MAX_B - MIN_B)
    
    return np.squeeze(np.expand_dims(data, axis=-1))
     
def denormalize(image):
    img = image.copy()
    MIN_B= -1024.0
    MAX_B= 3072.0

    return img * (MAX_B - MIN_B) + MIN_B

def trunc(mat):
    min = -160.0
    max = 240.0
    
    mat[mat <= min] = min
    mat[mat >= max] = max
    return mat
    
    
# A function that returns training images from the LowDoseCT Challenge dataset (link : https://www.aapm.org/grandchallenge/lowdosect/)
# If load_limited_images is True, it will load number of images that are specified in images_to_load.
# Else, the entire dataset will be loaded.
def load_training_images(number_of_images, patient_id, slice_type, low_dose_ct_training_dataset_dir='../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data'):
    print('Log : patient_id : ', patient_id, ' slice type : ', slice_type)
    
    training_filepaths_x = []   # i.e the QD (quarter dose) images (noisy images)
    training_filepaths_y = []   # i.e the FD (full dose) images (clean images)

    for root, folder_name, file_names in os.walk(os.path.join(low_dose_ct_training_dataset_dir, slice_type)):

        for file_name in file_names:
            file_path = os.path.join(root, file_name)
        
            if "QD" in file_name and patient_id in file_name:
                training_filepaths_x.append(file_path)
            elif "FD" in file_name and patient_id in file_name:
                training_filepaths_y.append(file_path)
            

    training_filepaths_x = np.unique(training_filepaths_x)
    training_filepaths_y = np.unique(training_filepaths_y)

    training_filepaths_x.sort()
    training_filepaths_y.sort()

    training_filepaths_x = training_filepaths_x[:number_of_images]
    training_filepaths_y = training_filepaths_y[:number_of_images]

    training_images_x = np.array([np.expand_dims(read_image(path), axis=-1) for path in training_filepaths_x])
    training_images_y = np.array([np.expand_dims(read_image(path), axis=-1) for path in training_filepaths_y])
        
    print('loaded training images x and y of len : ', len(training_images_x), len(training_images_y), ' respectively')
    print('type of train images x : ', training_images_x[0].dtype)
    print('range of values in train images : ', np.min(training_images_x[0]), np.max(training_images_x[0]))
    print('type of train images y : ', training_images_y[0].dtype)
    
    return training_images_x, training_images_y
