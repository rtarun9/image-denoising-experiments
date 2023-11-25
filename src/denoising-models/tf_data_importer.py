import numpy as np
import os
import tensorflow as tf
import pydicom

def read_image(image_path):
    data = pydicom.dcmread(image_path.numpy().decode('utf-8'))
    return np.expand_dims(np.expand_dims(data.pixel_array, axis=-1), axis=0)

class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, stride, name):
        super(PatchExtractor, self).__init__(name=name)
        self.patch_size = patch_size
        self.stride = stride

    def call(self, images):
        patch_depth = tf.shape(images)[-1]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [-1, self.patch_size, self.patch_size, patch_depth])
        return patches

def load_and_preprocess_image(file_path, patch_extractor=None):
    image_data = tf.py_function(read_image, [file_path], tf.float32)
    image_data = image_data / tf.reduce_max(image_data)
        
    if patch_extractor:
        patches = patch_extractor(image_data)
        return patches

    return image_data

def _load_image_paths(low_dose_ct_training_dataset_dir, load_limited_images, num_images_to_load):
    noisy_image_paths = []
    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            if "QD" in file_name:
                file_path = os.path.join(root, file_name)
                noisy_image_paths.append(file_path)

    clean_image_paths = []
    for root, folder_name, file_names in os.walk(low_dose_ct_training_dataset_dir):
        for file_name in file_names:
            if "FD" in file_name:   
                file_path = os.path.join(root, file_name)
                clean_image_paths.append(file_path)

    noisy_image_paths.sort()
    clean_image_paths.sort()

    if load_limited_images:
        noisy_image_paths = noisy_image_paths[:num_images_to_load]
        clean_image_paths = clean_image_paths[:num_images_to_load]
    
    return noisy_image_paths, clean_image_paths

def _create_image_dataset(image_paths, patch_extractor):
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    image_dataset = image_dataset.map(lambda file_path: load_and_preprocess_image(file_path, patch_extractor), num_parallel_calls=tf.data.AUTOTUNE)

    return image_dataset

def load_training_tf_dataset(low_dose_ct_training_dataset_dir='../../../Dataset/LowDoseCTGrandChallenge/Training_Image_Data', load_as_patches=False, load_limited_images=False, num_images_to_load=10):
    noisy_image_paths, clean_image_paths = _load_image_paths(low_dose_ct_training_dataset_dir, load_limited_images, num_images_to_load)

    patch_extractor = None
    if load_as_patches:
        patch_extractor = PatchExtractor(patch_size=64, stride=64, name="patch_extractor")

    noisy_image_dataset = _create_image_dataset(noisy_image_paths, patch_extractor)
    clean_image_dataset = _create_image_dataset(clean_image_paths, patch_extractor)

    training_dataset = tf.data.Dataset.zip((noisy_image_dataset, clean_image_dataset))

    return training_dataset
