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

def run_models(patient_id, slice_type):
    noisy_array, gt_array = load_training_images(patient_id, slice_type, '../../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data/')

    noisy_array = np.concatenate((noisy_array, _n), axis=0)
    gt_array = np.concatenate((gt_array, _g), axis=0)
