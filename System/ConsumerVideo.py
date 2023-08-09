import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import redis
import kafka
import time
from datetime import datetime
from threading import Event, Thread


class VideoStreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player App")
        self.video_source = 0

        self.canvas = tk.Canvas(root, width=1000, height=600)
        self.canvas.pack()

        self.btn_start_player = tk.Button(root, text="Start Player", command=self.start_player)
        self.btn_start_player.pack(pady=5)

        self.btn_stop_player = tk.Button(root, text="Stop Player", command=self.stop_player, state=tk.DISABLED)
        self.btn_stop_player.pack(pady=5)

        self.video_stream = None
        self.is_streaming = False
        self.event = Event()

        # Redis
        self.red = redis.Redis()

        # Kafka
        self.topic = 'frame_notification'
        self.consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest',
                                            group_id='grp_visualization', consumer_timeout_ms=2000)
        self.consumer.subscribe([self.topic])

        # Initialize FPS calculation variables
        self.start_time = time.time()
        self.frame_count = 0

    def start_player(self):
        self.is_streaming = True
        self.btn_start_player.config(state=tk.DISABLED)
        self.btn_stop_player.config(state=tk.NORMAL)
        self.video_stream = cv2.VideoCapture(self.video_source)

        # Start the consumer thread to process video frames
        consumer_thread = Thread(target=self.consume_video_frames, args=(self.consumer,))
        consumer_thread.start()

    def stop_player(self):
        self.is_streaming = False
        self.btn_start_player.config(state=tk.NORMAL)
        self.btn_stop_player.config(state=tk.DISABLED)
        self.event.set()  # Set the event to stop the consumer thread
        if self.video_stream:
            self.video_stream.release()

    def consume_video_frames(self, consumer):
        while not self.event.is_set():
            for message in consumer:
                print("Hi2")
                if message.value.decode("utf-8") == "new_frame":
                    frame_time = datetime.fromtimestamp(message.timestamp / 1000)
                    curr_time = datetime.now()
                    diff = (curr_time - frame_time).total_seconds()

                    # Exclude old frames
                    if diff < 2:
                        frame_temp = np.frombuffer(self.red.get("frame:latest"), dtype=np.uint8)

                        if np.shape(frame_temp)[0] != 0:
                            frame = frame_temp.reshape((480, 640, 3))
                            self.display_frame(frame)

                            # Calculate FPS
                            self.frame_count += 1
                            elapsed_time = time.time() - self.start_time
                            fps = self.frame_count / elapsed_time
                            print("FPS: " + str(fps))

                if self.event.is_set():
                    break

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.photo = photo


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStreamApp(root)
    root.mainloop()
