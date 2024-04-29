import tkinter as tk

# Window setup.
window = tk.Tk()
window.title("LDCT Visualizer")
window.geometry("1280x720")

# Top title.
title_frame = tk.Frame(window)
title_frame.pack(side=tk.TOP, fill=tk.X)

main_title = tk.Label(title_frame, text="LDCT Visualizer", font=("Arial", 18))
main_title.pack(padx=5, pady=5)

# Left frame.
left_frame = tk.Frame(window)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Patient ID list setup.
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

patient_id_label = tk.Label(left_frame, text="Patient ID:")
patient_id_label.pack()

patient_id_widget = tk.StringVar(left_frame)
patient_id_widget.set(patient_ids[0])

patient_option_menu = tk.OptionMenu(left_frame, patient_id_widget, *patient_ids)
patient_option_menu.pack()

# Slice type option.
slice_types= [
    '1mm B30',
    '1mm B45',
    '3mm B30',
    '3mm B45',
]

slice_type_label = tk.Label(left_frame, text="Slice Types:")
slice_type_label.pack()

slice_type_widget = tk.StringVar(left_frame)
slice_type_widget.set(slice_types[0])

slice_type_option_menu = tk.OptionMenu(left_frame, slice_type_widget, *slice_types)
slice_type_option_menu.pack()

window.mainloop()
