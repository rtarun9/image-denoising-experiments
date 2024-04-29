import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from model_runner import run_models

dpg.create_context()

def save_init():
    dpg.save_init_file("dpg.ini")

dpg.configure_app(init_file="dpg.ini")  

selected_patient_id = ""
selected_slice_type = ""

def get_selected_item(sender, app_data, user_data):
    print(f"sender: {sender}, \t app_data: {app_data}, \t user_data: {user_data}")

    global selected_patient_id
    global selected_slice_type

    # All patient ID's start with L.
    if app_data[0]== 'L':
        selected_patient_id = dpg.get_value(sender)
        print('selected patient id : ', selected_patient_id)
    else:
        selected_slice_type = dpg.get_value(sender)
        print('selected slice type: ', selected_slice_type)

fd_array = []

texture_data = []
for i in range(0, 512* 512):
    texture_data.append(255 / 255)
    texture_data.append(0)
    texture_data.append(255 / 255)
    texture_data.append(255 / 255)

with dpg.texture_registry(show=True):
    dpg.add_raw_texture(width=512, height=512, default_value=np.array(texture_data), format=dpg.mvFormat_Float_rgba, tag="texture_tag")


def run_models_and_get_outputs(sender, app_data, user_data):
    print('run models with output')
    global fd_array

    fd_array = run_models(selected_patient_id, selected_slice_type)

    print('len fd array : ', len(fd_array))
    print(fd_array.shape)
    texture_data = fd_array[0]

with dpg.window(label="Config Menu", tag="config_menu"):

    # Patient ID
    dpg.add_text('Patient ID')
    patient_ids = [
        'L067',
        'L096',
        'L109',
        'L143',
        'L192',
        'L286',
        'L291',
        'L310',
        'L333',
        'L506'
    ]
    dpg.add_combo(patient_ids,  callback=get_selected_item)

    # Slice Type
    dpg.add_text('Slice Type')
    slice_type= [
        '1mm B30',
        '1mm D45',
        '3mm B30',
        '3mm D45'
    ]
    dpg.add_combo(slice_type,  callback=get_selected_item)

    dpg.add_button(label="Run Visualizer", callback=lambda: run_models_and_get_outputs(None, None, None))

    dpg.add_button(label="Save UI config", callback=lambda: save_init)

dpg.create_viewport(title='LDCT Visualizer', width=1280, height=720)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()