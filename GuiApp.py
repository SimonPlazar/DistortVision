import os
import tkinter as tk
import cv2
import cvzone
import numpy as np
import torch
from tkinter import ttk
from ultralytics import YOLO
from PIL import Image

def get_device_id_by_name(device_name):
    if device_name == 'CPU':
        return "cpu"

    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_name(i) == device_name:
            return i

    return None  # Return None if the device name is not found
class VideoProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing GUI")

        self.resolutions = ["1920x1080", "1280x720", "640x360"]
        self.distortion_options = ["None", "Jpeg", "Censoring", "Pink"]

        self.create_widgets()

    def create_widgets(self):
        # Video Resolution Selection
        resolution_label = tk.Label(self.root, text="Select Video Resolution:")
        resolution_label.pack(pady=10)

        self.resolution_var = tk.StringVar()
        self.resolution_combobox = ttk.Combobox(self.root, textvariable=self.resolution_var, values=self.resolutions)
        self.resolution_combobox.pack()

        # CUDA Torch Device Selection
        device_label = tk.Label(self.root, text="Select Torch Device:")
        device_label.pack(pady=10)

        self.device_var = tk.StringVar()
        self.device_combobox = ttk.Combobox(self.root, textvariable=self.device_var)
        self.update_device_options()
        self.device_combobox.pack()

        # Distortion Options
        distortion_label = tk.Label(self.root, text="Select Distortion Option:")
        distortion_label.pack(pady=10)

        self.distortion_var = tk.StringVar()
        self.distortion_combobox = ttk.Combobox(self.root, textvariable=self.distortion_var, values=self.distortion_options)
        self.distortion_combobox.pack()

        # YOLO Weights Selection
        weights_label = tk.Label(self.root, text="Select YOLO Weights:")
        weights_label.pack(pady=10)

        self.weights_var = tk.StringVar()
        self.weights_combobox = ttk.Combobox(self.root, textvariable=self.weights_var)
        self.update_weights_options()
        self.weights_combobox.pack()

        # Apply Button
        self.apply_button = tk.Button(self.root, text="Start", command=self.apply_settings)
        self.apply_button.pack(pady=20)

    def update_device_options(self):
        available_devices = ['CPU']
        if torch.cuda.is_available():
            available_devices += [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        self.device_combobox['values'] = available_devices

    def update_weights_options(self):
        available_weights = []
        for root, dirs, files in os.walk("weights"):
            for file in files:
                if file.endswith(".pt") or file.endswith(".pth"):
                    available_weights.append(os.path.join(root, file))

        self.weights_combobox['values'] = available_weights

    def apply_settings(self):
        if not bool(self.resolution_var.get()) or not bool(self.device_var.get()) or not bool(self.distortion_var.get()) or not bool(self.weights_var.get()):
            print("Error: Not all settings selected")
            return

        self.root.destroy()  # Close the window


root = tk.Tk()
app = VideoProcessingApp(root)
root.geometry("300x330")  # Set window size
root.mainloop()

if not bool(app.resolution_var.get()) or not bool(app.device_var.get()) or not bool(app.distortion_var.get()) or not bool(app.weights_var.get()):
    exit(0)

width, height = app.resolution_var.get().split('x')
width, height = int(width), int(height)

device_id = get_device_id_by_name(app.device_var.get())

path_to_weights = app.weights_var.get()

# DEBUG PRINTS
# print("Resolution: {} x {}".format(width, height))
# print("Device: {} ({})".format(app.device_var.get(), device_id))
# print("Distortion: {}".format(app.distortion_var.get()))
# print("Path to Weights: {}".format(path_to_weights))

def censoring(extracted):
    image = Image.fromarray(cv2.cvtColor(cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV), cv2.COLOR_BGR2RGB))
    pixilated = image.resize((image.width // 7, image.height // 7), resample=Image.NEAREST)
    pixilated = pixilated.resize(image.size, resample=Image.NEAREST)
    return cv2.cvtColor(np.array(pixilated), cv2.COLOR_RGB2BGR)

def jpeg_artifact(extracted):
    _, compressed_image = cv2.imencode(".jpg", cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV), [int(cv2.IMWRITE_JPEG_QUALITY), 6])
    return cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

# Load the model
model = YOLO(path_to_weights)

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

capture.set(cv2.CAP_PROP_FPS, 60)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fpsReader = cvzone.FPS()

none = False

if app.distortion_var.get() == "None":
    none = True

while True:
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)

    if not none:
        # Get the masks from the model
        results = model.predict(source=frame, conf=0.59, classes=0, verbose=False, device=device_id, retina_masks=True)[0].masks

        if results is not None:  # If a mask is found
            mask = (results.data[0] * 255).to(torch.uint8)

            for i in range(1, results.data.shape[0]):  # For each other mask
                mask = torch.bitwise_or(mask, (results.data[i] * 255).to(torch.uint8))  # Combine the masks

            # Pass the mask to cpu memory
            mask = mask.cpu().numpy()

            # Resize the mask to the video stream size
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create dilated mask and extract the area defined by the mask from the original image
            dilated_mask = cv2.dilate(mask, None, iterations=5)
            extracted = cv2.bitwise_and(frame, frame, mask=dilated_mask)

            if app.distortion_var.get() == "Censoring":
                extracted = censoring(extracted)
            elif app.distortion_var.get() == "Jpeg":
                extracted = jpeg_artifact(extracted)
            elif app.distortion_var.get() == "Pink":
                extracted = np.full((height, width, 3), (203, 192, 255), dtype=np.uint8)

            # Use bitwise operations to replace the pixels in the original image
            roi = cv2.bitwise_and(extracted, extracted, mask=mask)
            inverse_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

            frame = cv2.add(background, roi)
        else: # If no mask is found
            cv2.putText(frame, "No mask found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Display a message

    fpsReader.update(frame, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()