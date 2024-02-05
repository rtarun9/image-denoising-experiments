import pydicom
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import h5py

import os

def load_scan(data):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    return data


def get_pixels_hu(image):
    # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    image = image.astype(np.int16)
    return np.int16(image)

def read_image(data):
    full_pixels =get_pixels_hu(load_scan(data))
    

    
    data = np.expand_dims(data, axis=-1)
    data = cv2.resize(data, (512, 512))
    data = np.expand_dims(data, -1)

    return data
     

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
    
    
    return np.squeeze(np.expand_dims(data, axis=-1), axis=0).astype(np.float32)

def load_image_paths(lodopab_ct_training_dataset_dir,  validation_patient_number="L506"): 
    noisy_image_paths = []
    for root, folder_name, file_names in os.walk(lodopab_ct_training_dataset_dir):
        for file_name in file_names:
                file_path = os.path.join(root, file_name)
                noisy_image_paths.append(file_path)

    
    return noisy_image_paths

def patch_extractor(image, patch_width=64, patch_height=64):
    image = image.copy()
    
    image_height, image_width, channels = image.shape
    patches = image.reshape(-1, image_height // patch_height, patch_height, image_width // patch_width, patch_width, channels)
    patches = patches.swapaxes(2, 3)
    patches = patches.reshape(-1, patch_height, patch_width, channels)
    
    return torch.from_numpy(patches)
     
class LodopabCTDataset(Dataset):
    def __init__(self, root_dataset_dir):
        self.root_dataset_dir = root_dataset_dir
        self.noisy_image_paths= load_image_paths(root_dataset_dir) 

        print("number of image paths : ", len(self.noisy_image_paths))
        
    def __len__(self):
        return len(self.noisy_image_paths) 
    
    def __getitem__(self, idx):
        noisy_image_path = self.noisy_image_paths[idx]
        noisy_image_data = h5py.File(noisy_image_path)

        group_key = list(noisy_image_data.keys())[0]

        noisy_data = list(noisy_image_data[group_key])
        data = [None] * len(noisy_data)
        for i in range(len(noisy_data)):
            noisy_image = patch_extractor(read_image(noisy_data[i]))
            data[i] = noisy_image

        return data 
     
def get_validation_dataloader(root_dataset_dir, shuffle=True):

    validation_data = LodopabCTDataset(root_dataset_dir)    
 
    print(f"validation data image len : , {len(validation_data)}")
     
    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=shuffle)
    
    return validation_dataloader
