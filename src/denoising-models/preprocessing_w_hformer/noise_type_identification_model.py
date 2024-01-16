#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os


import tensorflow as tf
from tensorflow.keras import layers, models


def get_noise_type_identification_model():
    model = models.Sequential()

    model.add(layers.Input(shape=(64, 64, 1)))

    model.add(layers.Conv2D(5, (3, 3),  padding='same'))
    model.add(layers.Dropout(0.1)) 

    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(layers.Conv2D(50, (8, 8),  padding='same'))
    model.add(layers.Dropout(0.1)) 

    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(4, activation='softmax'))
 
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('weights/saved_noise_type_model.h5')
    
    return model

def denoise_image(model, images):
    #print('before denoising, shape of images is : ', images.shape)
    #print('shape of each image : ', images[0].shape)
    noise_types = model.predict(images.cpu().numpy())
    images = images.cpu().numpy()
    num_images = images.shape[0]
    
    denoised_images = [None] * num_images
   
    NOISE_TYPE_IMPULSE = 0
    NOISE_TYPE_GAUSSIAN = 1
    NOISE_TYPE_SPECKLE = 2
    NOISE_TYPE_POISSON = 3
     
    for i in range(num_images):
        noise_type = np.argmax(noise_types[i])
        #print(noise_type)
        if (noise_type == NOISE_TYPE_IMPULSE):
            image_after_median_filter = cv2.medianBlur(images[i], 3)
            denoised_images[i] = image_after_median_filter
        elif (noise_type == NOISE_TYPE_GAUSSIAN):
            image_after_gaussian_blur = cv2.GaussianBlur(images[i], (3, 3), 0)
            denoised_images[i] = image_after_gaussian_blur
        elif (noise_type == NOISE_TYPE_SPECKLE):
            image_after_median_filter = cv2.medianBlur(images[i], 3)
            denoised_images[i] = image_after_median_filter
        elif (noise_type == NOISE_TYPE_POISSON):
            image_after_nlm = cv2.fastNlMeansDenoising((images[i] * 255).astype(np.uint8), None, 10, 5, 15) 
            denoised_images[i] = image_after_nlm
            

        #plt.imshow(images[i], 'gray')
        #plt.show()
        #plt.imshow(denoised_images[i], 'gray')
        #plt.show() 
    return denoised_images
 
        
            
        