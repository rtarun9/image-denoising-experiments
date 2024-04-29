# Given a slice type and patient name, return a list of:
# QD list, FD list, Hformer list, W emd list.

import warnings
warnings.filterwarnings("ignore")

import os
import pathlib
from matplotlib import pyplot as plt
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

def calculate_psnr(original_image, reconstructed_image,range=400):
    return peak_signal_noise_ratio(original_image, reconstructed_image,data_range=range) 

    psnr_value = peak_signal_noise_ratio(original_image, reconstructed_image, data_range=240+160)
    return psnr_value

def calculate_ssim(original_image, reconstructed_image, range=400.0):    
    ssim_value = ssim(original_image.astype(np.int16), reconstructed_image.astype(np.int16), win_size=11, channel_axis=2, data_range=range)
    return ssim_value

def calculate_rmse(original_image, reconstructed_image):
    return mse(original_image, reconstructed_image)

def get_average_metrics(predicted_images, _gt_array, _noisy_array):
    psnr_original_mean = 0
    psnr_prediction_mean = 0

    ssim_original_mean = 0
    ssim_prediction_mean = 0

    mse_original_mean = 0
    mse_prediction_mean = 0

    varience_of_laplacian = 0

    if np.all(_gt_array) != None:
        gt_array = _gt_array
        noisy_array = _noisy_array
        
    kernel = np.array([[0, -1, 0],
                    [-1, -8, -1],
                    [0, -1, 0]], dtype=np.float32)

    i = 0
    for gt_img, noisy_img, predicted_img in zip(gt_array, noisy_array, predicted_images):
        predicted_img=  predicted_images[i]
            
        copy_pred_img = np.squeeze(predicted_img, axis=-1)
        print(copy_pred_img.dtype, copy_pred_img.shape)
        ddtype = None
        if copy_pred_img.dtype == np.float64:
            ddtype = cv2.CV_64F
        else:
            ddtype = cv2.CV_32F

        varience_of_laplacian += cv2.filter2D(copy_pred_img, ddtype, kernel).var()

        psnr_recon =  calculate_psnr(trunc(denormalize(gt_img)), trunc(denormalize(predicted_img)))
        psnr_qd =  calculate_psnr(trunc(denormalize(gt_img)),  trunc(denormalize(noisy_img)))
        ssim_recon = calculate_ssim(trunc(denormalize(gt_img)),  trunc(denormalize(predicted_img)))
        ssim_qd =calculate_ssim(trunc(denormalize(gt_img)), trunc(denormalize(noisy_img)))
        rmse_recon = calculate_rmse(trunc(denormalize(gt_img)),  trunc(denormalize(predicted_img)))
        rmse_qd=calculate_rmse(trunc(denormalize(gt_img)), trunc(denormalize(noisy_img)))

        psnr_original_mean += psnr_qd
        psnr_prediction_mean += psnr_recon
        
        ssim_original_mean += ssim_qd
        ssim_prediction_mean += ssim_recon

        mse_original_mean += rmse_qd
        mse_prediction_mean += rmse_recon
        
        i = i + 1        
    
    psnr_original_mean/=gt_array.shape[0]
    psnr_prediction_mean/=gt_array.shape[0]

    ssim_original_mean/=gt_array.shape[0]
    ssim_prediction_mean/=gt_array.shape[0]

    mse_original_mean/=gt_array.shape[0]
    mse_prediction_mean/=gt_array.shape[0]
    
    print("Original average gt-noisy PSNR ->", psnr_original_mean)
    print("Predicted average gt-predicted PSNR ->", psnr_prediction_mean)

    print("Original average gt-noisy SSIM ->", ssim_original_mean)
    print("Predicted average gt-predicted SSIM ->", ssim_prediction_mean)

    print("Original average gt-noisy MSE->", mse_original_mean)
    print("Predicted average gt-predicted MSE->", mse_prediction_mean)

    print('VAL : ', varience_of_laplacian)
    
    return round(psnr_prediction_mean, 4), round(ssim_prediction_mean, 4), round(mse_prediction_mean, 4), round(psnr_prediction_mean - psnr_original_mean, 4), round(ssim_prediction_mean - ssim_original_mean, 4), round(mse_prediction_mean - mse_original_mean, 4), varience_of_laplacian

def get_hformer():
    import sys
    sys.path.append('../denoising-models/hformer_vit/model/')
    sys.path.append('../denoising-models/hformer_vit/')
    from hformer_model_extended import get_hformer_model, PatchExtractor

    hformer_model = get_hformer_model(num_channels_to_be_generated=64, name="hformer_model_extended")
    hformer_model.build(input_shape=(None, 64, 64, 1))
    hformer_model.load_weights('../denoising-models/hformer_vit/test/experiments/full_dataset/hformer_64_channel_custom_loss_epochs_48.h5')

    return hformer_model

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

def run_models(patient_id, slice_type):
    print('run models : ', patient_id, slice_type)
    noisy_array, gt_array = load_training_images(patient_id, slice_type, '../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data/')
    return noisy_array

    patch_extractor = PatchExtractor(patch_size=64, stride=64, name="patch_extractor")
    noisy_image_patches_array = patch_extractor(noisy_array)

    hformer_model = get_hformer()

    hformer_prediction_patches = hformer_model.predict(noisy_image_patches_array)

    hformer_predictions = np.expand_dims(reconstruct_image_from_patches(hformer_prediction_patches[0:64], 8), axis=0)

    for i in range(1, int(hformer_prediction_patches.shape[0] / 64)): 
        reconstructed_image = reconstruct_image_from_patches(hformer_prediction_patches[i * 64 : i * 64 + 64], num_patches_per_row=8)
        reconstructed_image = np.expand_dims(reconstructed_image, axis=0)

        hformer_predictions = np.append(hformer_predictions, reconstructed_image, axis=0)

    return hformer_predictions
