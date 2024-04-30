import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt

import numpy as np

from model_runner import run_models

# Globla variables
selected_patient_id = ""
selected_slice_type = ""
selected_number_of_images = 1

model_outputs = []

# Dpg setup.
dpg.create_context()

def save_init():
    dpg.save_init_file("dpg.ini")

dpg.configure_app(init_file="dpg.ini")  


image_slider_number = 0
def image_slider_get_selected_item(sender, app_data, user_data):
    global image_slider_number 

    image_slider_number = app_data
    print(app_data)
    print(image_slider_number)
    update_texture(None, None, None)

def get_selected_item(sender, app_data, user_data):
    global selected_patient_id, selected_slice_type, selected_number_of_images

    if sender == patient_combo:
        selected_patient_id = app_data
    elif sender == slice_combo:
        selected_slice_type = app_data
    else:
        selected_number_of_images = int(app_data)


def run_models_and_get_outputs(sender, app_data, user_data):
    global model_outputs 

    model_outputs = run_models(selected_patient_id, selected_slice_type, selected_number_of_images)
    update_texture(None, None, None)

texture_data = []
for i in range(0, 512* 512 * 2):
    texture_data.append(255 / 255)
    texture_data.append(1)
    texture_data.append(255 / 255)
    texture_data.append(255 / 255)

with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width=1024, height=512, default_value=texture_data, tag="texture_tag")


def update_texture(sender, app_data, user_data):
    global image_slider_number 
    if len(model_outputs) > 0:
        if image_slider_number >= len(model_outputs):
            image_slider_number = len(model_outputs) - 1
            
        new_texture_source_data = model_outputs[image_slider_number]

        dpg.set_value("texture_tag", new_texture_source_data)


with dpg.window(label="Texture"):
    dpg.add_image("texture_tag")

with dpg.window(label="Config Menu", tag="config_menu"):
    dpg.add_text('Patient ID')
    patient_combo = dpg.add_combo(["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"], callback=get_selected_item)

    dpg.add_text('Slice Type')
    slice_combo = dpg.add_combo(["1mm B30", "1mm D45", "3mm B30", "3mm D45"], callback=get_selected_item)

    dpg.add_text('Number of images')
    dpg.add_slider_int(default_value=10, tag="slider_int", callback=get_selected_item)

    dpg.add_button(label="Run Visualizer", callback=run_models_and_get_outputs)
    dpg.add_button(label="Update Texture", callback=update_texture)

    dpg.add_button(label="Save UI config", callback=save_init)

with dpg.window(label="Image Slider", tag="app_config"):
    dpg.add_slider_int(default_value=0, tag="image_slider", callback=image_slider_get_selected_item)
    
dpg.create_viewport(title='LDCT Visualizer', width=1920, height=1080)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()