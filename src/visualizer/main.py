import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt

import numpy as np

from model_runner import run_models

# Globla variables
selected_patient_id = ""
selected_slice_type = ""

model_outputs = []

# Dpg setup.
dpg.create_context()

def save_init():
    dpg.save_init_file("dpg.ini")

dpg.configure_app(init_file="dpg.ini")  


def get_selected_item(sender, app_data, user_data):
    global selected_patient_id, selected_slice_type

    if sender == patient_combo:
        selected_patient_id = app_data
    elif sender == slice_combo:
        selected_slice_type = app_data


def run_models_and_get_outputs(sender, app_data, user_data):
    global model_outputs 

    model_outputs = run_models(selected_patient_id, selected_slice_type)

texture_data = []
for i in range(0, 512* 512):
    texture_data.append(255 / 255)
    texture_data.append(1)
    texture_data.append(255 / 255)
    texture_data.append(255 / 255)

with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width=512, height=512, default_value=texture_data, tag="texture_tag")


def update_texture(sender, app_data, user_data):
    new_color = dpg.get_value(sender)
    new_color[0] = new_color[0] / 255
    new_color[1] = new_color[1] / 255
    new_color[2] = new_color[2] / 255
    new_color[3] = new_color[3] / 255
    print(new_color)

    new_texture_data = []
    for i in range(0, 512 * 512):
        new_texture_data.append(new_color[0])
        new_texture_data.append(new_color[1])
        new_texture_data.append(new_color[2])
        new_texture_data.append(new_color[3])

    dpg.set_value("texture_tag", new_texture_data)


with dpg.window(label="Tutorial"):
    dpg.add_image("texture_tag")
    dpg.add_color_picker((255, 0, 255, 255), label="Texture",
                         no_side_preview=True, alpha_bar=True, width=200,
                         callback=update_texture)

with dpg.window(label="Config Menu", tag="config_menu"):
    dpg.add_text('Patient ID')
    patient_combo = dpg.add_combo(["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"], callback=get_selected_item)

    dpg.add_text('Slice Type')
    slice_combo = dpg.add_combo(["1mm B30", "1mm D45", "3mm B30", "3mm D45"], callback=get_selected_item)

    dpg.add_button(label="Run Visualizer", callback=run_models_and_get_outputs)
    dpg.add_button(label="Update Texture", callback=update_texture)

    dpg.add_button(label="Save UI config", callback=save_init)

dpg.create_viewport(title='LDCT Visualizer', width=1280, height=720)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()