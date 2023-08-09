import numpy as np
import cv2
import threading
import redis
import signal
import time
import kafka

def thread_produce():
    # Redis
    red = redis.Redis()

    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
    vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = vc.get(cv2.CAP_PROP_FPS)
    width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(fps, width, height)

    # Kafka
    topic = 'frame_noticifation'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

    while True:
        t_start = time.perf_counter()
        ret, frame = vc.read()

        # Jump back to the beginning of input
        # if not ret:
        #     vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Add frame to redis
        red.set("frame:latest", np.array(frame).tobytes())

        # Send notification about new frame over Kafka
        future = producer.send(topic, b"new_frame", timestamp_ms=round(time.time() * 1000))

        # Wait until message is delivered to Kafka
        try:
            rm = future.get(timeout=10)
        except kafka.KafkaError:
            pass

        # # Preserve FPS
        # t_stop = time.perf_counter()
        # t_elapsed = t_stop - t_start
        # t_frame = 1000 / fps / 1000
        # t_sleep = t_frame - t_elapsed
        # if t_sleep > 0:
        #     time.sleep(t_sleep)

        # Stop loop
        if event.is_set():
            vc.release()
            break


def sigint_handler(signum, frame):
    event.set()
    thread.join()
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

event = threading.Event()
thread = threading.Thread(target=lambda: thread_produce())

if __name__ == "__main__":
    thread.start()
    input("Press CTRL+C or Enter to stop producing...")
    event.set()
    thread.join()