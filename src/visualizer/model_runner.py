# Given a slice type and patient name, return a list of:
# QD list, FD list, Hformer list, W emd list.

from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import os
import pathlib
import torch
import numpy as np
import cv2
from pyemd.EMD2d import EMD2D
emd2d = EMD2D()

import random

import data_importer

from data_importer import load_training_images
from data_importer import trunc, denormalize

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import sys

from skimage.metrics import peak_signal_noise_ratio

from skimage.metrics import mean_squared_error  as mse

import sys
sys.path.append('../denoising-models/hformer_vit/model/')
sys.path.append('../denoising-models/hformer_vit/')

from hformer_model_extended import  PatchExtractor

def reconstruct_image_from_patches(patches, num_patches_per_row):
    patch_size = patches.shape[1]  # Assuming square patches
    num_patches = patches.shape[0]

    # Calculate the number of rows
    num_patches_per_col = num_patches // num_patches_per_row

    # Initialize an empty image to store the reconstructed result
    reconstructed_image = np.zeros((num_patches_per_col * patch_size, num_patches_per_row * patch_size))

    # Reshape the patches into a 2D array
    patches_2d = patches.reshape((num_patches_per_col, num_patches_per_row, patch_size, patch_size))
    # Reconstruct the image by placing each patch in its corresponding position

    for i in range(num_patches_per_col):
        for j in range(num_patches_per_row):
            reconstructed_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches_2d[i, j]

    return np.expand_dims(reconstructed_image, axis=-1)

def get_w_outputs(noisy_array, noisy_image_patches_array):
    sys.path.append('../denoising-models/hformer_pytorch')
    from torchinfo import summary

    from w_model import WModel 

    w_model = WModel(num_channels=32).cuda()
    w_model.load_state_dict(torch.load('../denoising-models/hformer_pytorch/weights/model_278.pth'))
    w_model.eval()

    w_prediction_patches = []

    with torch.no_grad():    
        for i, data in enumerate(noisy_image_patches_array):
            noisy = data
        
            predictions = w_model(torch.unsqueeze(torch.from_numpy(noisy.numpy()), dim=0).to('cuda')).cpu()

            w_prediction_patches.append(predictions.detach().cpu())
        
    w_prediction_patches = np.concatenate(w_prediction_patches, axis=0)

    w_predictions = np.expand_dims(reconstruct_image_from_patches(w_prediction_patches[0:64], 8), axis=0)


    for i in range(1, int(w_prediction_patches.shape[0] / 64)): 
        reconstructed_image = reconstruct_image_from_patches(w_prediction_patches[i * 64 : i * 64 + 64], num_patches_per_row=8)
        reconstructed_image = np.expand_dims(reconstructed_image, axis=0)

        w_predictions= np.append(w_predictions, reconstructed_image, axis=0)

    # In extended_w_predictions, do EMD

    emd_predictions = [None] * w_predictions.shape[0]
    for i in range(w_predictions.shape[0]):

        noisy_reshaped = np.squeeze(noisy_array[i], -1)
        noisy_imfs = emd2d.emd(noisy_reshaped,max_imf=-1)

        pred_reshaped = w_predictions[i]
        pred_reshaped = np.squeeze(pred_reshaped, axis=-1)
        pred_imfs = emd2d.emd(pred_reshaped, max_imf=-1)

        best_performing_lerp_image = None

        for x in [0.44]:
            swaped_IMFs = np.array([noisy_imfs[1] * x + pred_imfs[1] * (1.0 - x), pred_imfs[0] * (1.0 - x) + noisy_imfs[0] * x])
            predictions = torch.from_numpy(np.expand_dims(np.expand_dims(np.sum(swaped_IMFs, axis=0), -1), 0))

            best_performing_lerp_image = predictions
        print('w emd done for : ', i)

        w_predictions[i] = best_performing_lerp_image

    return w_predictions

def get_hformer_outputs(noisy_array):
    import sys
    sys.path.append('../denoising-models/hformer_vit/model/')
    sys.path.append('../denoising-models/hformer_vit/')
    from hformer_model_extended import get_hformer_model, PatchExtractor

    hformer_model = get_hformer_model(num_channels_to_be_generated=64, name="hformer_model_extended")
    hformer_model.build(input_shape=(None, 64, 64, 1))
    hformer_model.load_weights('../denoising-models/hformer_vit/test/experiments/full_dataset/hformer_64_channel_custom_loss_epochs_48.h5')

    patch_extractor = PatchExtractor(patch_size=64, stride=64, name="patch_extractor")
    noisy_image_patches_array = patch_extractor(noisy_array)

    hformer_prediction_patches = hformer_model.predict(noisy_image_patches_array)

    hformer_predictions = np.expand_dims(reconstruct_image_from_patches(hformer_prediction_patches[0:64], 8), axis=0)

    for i in range(1, int(hformer_prediction_patches.shape[0] / 64)): 
        reconstructed_image = reconstruct_image_from_patches(hformer_prediction_patches[i * 64 : i * 64 + 64], num_patches_per_row=8)
        reconstructed_image = np.expand_dims(reconstructed_image, axis=0)

        hformer_predictions = np.append(hformer_predictions, reconstructed_image, axis=0)

    return hformer_predictions


import tensorflow as tf
def run_models(patient_id, slice_type, number_of_images):
    print('run models : ', patient_id, slice_type)
    noisy_array, gt_array = load_training_images(number_of_images, patient_id, slice_type, '../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data/')

    patch_extractor = PatchExtractor(patch_size=64, stride=64, name="patch_extractor")
    noisy_image_patches_array = patch_extractor(noisy_array)

    with ThreadPoolExecutor() as executor:
        num_threads = executor._max_workers
        print(f"Number of threads: {num_threads}")

        hformer_future = executor.submit(get_hformer_outputs, tf.identity(noisy_image_patches_array))
        w_model_future = executor.submit(get_w_outputs, tf.identity(noisy_array), tf.identity(noisy_image_patches_array))

        # Wait for both futures to complete and retrieve results
        for future in as_completed([hformer_future, w_model_future]):
            if future == hformer_future:
                hformer_output = future.result()
            elif future == w_model_future:
                w_model_emd = future.result()

    output_plots = []
    for k in range(len(noisy_array)):
        fig, ax = plt.subplots(2, 3, figsize=((3 * 512)/72, 2 * 512/72), dpi=72)
        fig.tight_layout()

        ax[0,0].imshow(trunc(denormalize(noisy_array[k])), vmin=-160.0, vmax=240.0, cmap='gray')
        ax[0,0].axis('off')
        ax[0,0].set_title('Noisy Image')

        ax[0,1].imshow(trunc(denormalize(gt_array[k])), vmin=-160.0, vmax=240.0, cmap='gray')
        ax[0,1].axis('off')
        ax[0,1].set_title('Ground Truth')

        ax[0,2].imshow(trunc(denormalize(hformer_output[k])), vmin=-160.0, vmax=240.0, cmap='gray')
        ax[0,2].axis('off')
        ax[0,2].set_title('Hformer Output')

        ax[1,0].imshow(trunc(denormalize(w_model_emd[k])), vmin=-160.0, vmax=240.0, cmap='gray')
        ax[1,0].axis('off')
        ax[1,0].set_title('W Model EMD')

        ax[1,1].axis('off')
        ax[1,2].axis('off')

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape((512 * 3, 512 * 2, 3))

        new_texture_data = np.empty((512 * 3, 512 * 2, 4))
        new_texture_data[:, :, :3] = data / 255.0
        new_texture_data[:, :, 3] = 1.0

        output_plots.append(new_texture_data.flatten())

        print('Done for image index : ', k)
    return output_plots
