import pydicom
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

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
    full_pixels = get_pixels_hu(load_scan(image_path))
    
    MIN_B= -1024.0
    MAX_B= 3072.0
    data = (full_pixels - MIN_B) / (MAX_B - MIN_B)
    
    return np.squeeze(np.expand_dims(data, axis=-1), axis=0).astype(np.float32)

# Mode is either train or validation.
def load_image_paths(low_dose_ct_training_dataset_dir, mode, validation_patient_number="L506"): 
    noisy_image_paths = []
    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            if "QD" in file_name:
                if (mode == "train" and validation_patient_number not in file_name) or (mode == "validation" and validation_patient_number in file_name):
                    file_path = os.path.join(root, file_name)
                    noisy_image_paths.append(file_path)

    clean_image_paths = []
    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            if "FD" in file_name:   
                if (mode == "train" and validation_patient_number not in file_name) or (mode == "validation" and validation_patient_number in file_name):
                    file_path = os.path.join(root, file_name)
                    clean_image_paths.append(file_path)

    noisy_image_paths.sort()
    clean_image_paths.sort()

    
    return noisy_image_paths, clean_image_paths

def patch_extractor(image, patch_width=64, patch_height=64):
    image = image.copy()
    
    image_height, image_width, channels = image.shape
    patches = image.reshape(-1, image_height // patch_height, patch_height, image_width // patch_width, patch_width, channels)
    patches = patches.swapaxes(2, 3)
    patches = patches.reshape(-1, patch_height, patch_width, channels)
    
    return torch.from_numpy(patches)
     
class LDCTDataset(Dataset):
    def __init__(self, root_dataset_dir, mode):
        self.root_dataset_dir = root_dataset_dir
        self.noisy_image_paths, self.clean_image_paths = load_image_paths(root_dataset_dir, mode) 

        print("number of image paths : ", len(self.noisy_image_paths))
        
    def __len__(self):
        return len(self.noisy_image_paths) 
    
    def __getitem__(self, idx):
        noisy_image = patch_extractor(read_image(self.noisy_image_paths[idx]))
        clean_image = patch_extractor(read_image(self.clean_image_paths[idx]))

        return noisy_image, clean_image 
     
def get_train_and_validation_dataloader(root_dataset_dir, shuffle=True):

    train_data = LDCTDataset(root_dataset_dir, "train")
    validation_data = LDCTDataset(root_dataset_dir, "validation")    
 
    print(f"Train and validation data image len : {len(train_data)}, {len(validation_data)}")
     
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=shuffle)
    
    return train_dataloader, validation_dataloader