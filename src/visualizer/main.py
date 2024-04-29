import dearpygui.dearpygui as dpg

dpg.create_context()

def save_init():
    dpg.save_init_file("dpg.ini")

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
    dpg.add_combo(patient_ids, default_value=patient_ids[0])

    # Slice Type
    dpg.add_text('Slice Type')
    slice_type= [
        '1mm B30',
        '1mm D45',
        '3mm B30',
        '3mm D45'
    ]
    dpg.add_combo(slice_type, default_value=slice_type[0])

    dpg.add_button(label="Save UI config", callback=lambda: save_init)


dpg.create_viewport(title='LDCT Visualizer', width=1280, height=720)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()