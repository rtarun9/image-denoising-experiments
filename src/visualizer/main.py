import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from model_runner import run_models

dpg.create_context()

def save_init():
    dpg.save_init_file("dpg.ini")

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

dpg.configure_app(init_file="dpg.ini")  
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
    dpg.add_combo(patient_ids, default_value=patient_ids[0], callback=get_selected_item)

    # Slice Type
    dpg.add_text('Slice Type')
    slice_type= [
        '1mm B30',
        '1mm D45',
        '3mm B30',
        '3mm D45'
    ]
    dpg.add_combo(slice_type, default_value=slice_type[0], callback=get_selected_item)

    #dpg.add_button(label="Run Visualizer", callback=lamba: run_models, )

    dpg.add_button(label="Save UI config", callback=lambda: save_init)

with dpg.window(label="QD", tag="qd_window"):
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    image = np.asarray(buf)
    image = image.astype(np.float32) / 255

    with dpg.texture_registry():
        dpg.add_raw_texture(
            250, 250, image, format=dpg.mvFormat_Float_rgba, tag="texture_id"
        )

    dpg.add_image("texture_id")



dpg.create_viewport(title='LDCT Visualizer', width=1280, height=720)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()