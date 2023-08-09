import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import threading
import redis
import time
import kafka

def thread_produce(event, source):

    # Redis
    red = redis.Redis()

    # Kafka
    topic = 'frame_noticifation'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

    if source == 0:
        camera = True
        vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        camera = False
        vc = cv2.VideoCapture(source)

    # vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    fps = vc.get(cv2.CAP_PROP_FPS)
    width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(fps, width, height)

    while True:
        t_start = time.perf_counter()
        ret, frame = vc.read()

        # Jump back to the beginning of input
        if not camera and not ret:
            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = vc.read()

        # Add frame to redis
        red.set("frame:latest", np.array(frame).tobytes())

        # Send notification about new frame over Kafka
        future = producer.send(topic, b"new_frame", timestamp_ms=round(time.time() * 1000))

        # Wait until message is delivered to Kafka
        try:
            rm = future.get(timeout=10)
        except kafka.KafkaError:
            pass

        if not camera:
            # Preserve FPS
            t_stop = time.perf_counter()
            t_elapsed = t_stop - t_start
            t_frame = 1000 / fps / 1000
            t_sleep = t_frame - t_elapsed
            if t_sleep > 0:
                time.sleep(t_sleep)

        # Stop loop
        if event.is_set():
            vc.release()
            break

class VideoStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Streaming App")
        self.video_source = 0

        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()

        self.source_slider = tk.Scale(root, from_=0, to=1, orient=tk.HORIZONTAL, resolution=1, length=100,
                                      command=self.select_source,
                                      showvalue=0)

        self.source_slider.pack(pady=10)

        self.filename_label = tk.Label(root, text="Selected: Camera")
        self.filename_label.pack(pady=5)

        self.btn_select_file = tk.Button(root, text="Select File", command=self.select_file)
        self.btn_select_file.pack(pady=5)
        self.btn_select_file.pack_forget()  # Hide the button

        self.btn_start_stream = tk.Button(root, text="Start Stream", command=self.start_stream)
        self.btn_start_stream.pack(pady=5)

        self.btn_stop_stream = tk.Button(root, text="Stop Stream", command=self.stop_stream, state=tk.DISABLED)
        self.btn_stop_stream.pack(pady=5)

    def select_source(self, value):
        self.video_source = int(value)
        if self.video_source == 0:  # Camera
            print("Camera")
            self.filename_label.config(text="Selected: Camera")
            self.video_source = 0
            self.btn_select_file.pack_forget()  # Hide the button
        else:  # Video file
            print("Video file")
            self.filename_label.config(text="Selected: Video file")
            self.btn_select_file.pack()  # Show the button

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")])
        if file_path:
            self.filename_label.config(text=f"Selected File: {file_path}")
            self.video_source = file_path

    def start_stream(self):
        self.is_streaming = True
        self.btn_start_stream.config(state=tk.DISABLED)
        self.btn_stop_stream.config(state=tk.NORMAL)

        self.event = threading.Event()
        self.thread = threading.Thread(target=lambda: thread_produce(self.event, self.video_source))
        self.thread.start()

    def stop_stream(self):
        self.is_streaming = False
        self.btn_start_stream.config(state=tk.NORMAL)
        self.btn_stop_stream.config(state=tk.DISABLED)

        self.event.set()
        self.thread.join()

    def __del__(self):
        if hasattr(self, "event"):
            self.event.set()
            self.thread.join()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStreamApp(root)
    root.mainloop()
