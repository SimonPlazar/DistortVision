import time

import numpy as np
import cv2
import threading
import redis
import kafka
import signal
from datetime import datetime

def thread_frames():
    # Redis
    red = redis.Redis()

    # Video
    frame = 0

    # Kafka
    topic = 'frame_noticifation'
    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest',
                                   group_id='grp_visualization', consumer_timeout_ms=2000)
    consumer.subscribe([topic])

    # Initialize FPS calculation variables
    start_time = time.time()
    frame_count = 0

    while True:
        # Read from Redis when message is received over Kafka
        for message in consumer:

            if message.value.decode("utf-8") == "new_frame":
                frame_time = datetime.fromtimestamp(message.timestamp / 1000)
                curr_time = datetime.now()
                diff = (curr_time - frame_time).total_seconds()

                # Exclude old frames
                if diff < 2:
                    frame_temp = np.frombuffer(red.get("frame:latest"), dtype=np.uint8)

                    # Convert image
                    # if (np.shape(frame_temp)[0] == 921600):
                    # if (np.shape(frame_temp)[0] == 6220800):
                    if (np.shape(frame_temp)[0] != 0):
                        # frame = frame_temp.reshape((480, 640, 3))
                        # frame = frame_temp.reshape((1080, 1920, 3))
                        frame = frame_temp.reshape((720, 1280, 3))
                        # frame = frame_temp.reshape((480, 640, 3))
                        # frame = frame_temp.reshape((1080, 1920, 3))
                        frame = frame_temp.reshape((720, 1280, 3))


                    # Display image
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)

                    # Calculate FPS
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    print("FPS: " + str(fps))


            if event.is_set():
                break

                # Stop loop
        if event.is_set():
            cv2.destroyAllWindows()
            break


def sigint_handler(signum, frame):
    event.set()
    thread_frm.join()
    exit(0)


signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread_frm = threading.Thread(target=lambda: thread_frames())

if __name__ == "__main__":
    thread_frm.start()
    input("Press CTRL+C or Enter to stop visualization...")
    event.set()
    thread_frm.join()