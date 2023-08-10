import ffmpeg
import numpy as np
import cv2
import threading
import redis
import signal
import time
import kafka
import cvzone


def thread_produce():
    # Redis
    red = redis.Redis()

    # Kafka
    topic = 'frame_noticifation'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

    input_device = "video=HD Pro Webcam C920"  # Use the camera name here

    ffmpeg_cmd = (
        ffmpeg.input(input_device, format='dshow')
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )

    fpsReader = cvzone.FPS()

    while True:
        in_bytes = ffmpeg_cmd.stdout.read(640 * 480 * 3)
        if not in_bytes:
            break

        frame = (
            np.frombuffer(in_bytes, np.uint8)
            .reshape([480, 640, 3])
        )

        # Add frame to redis
        # fpsReader.update(frame, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
        red.set("frame:latest", np.array(frame[:, :, [2, 1, 0]]).tobytes())

        # Send notification about new frame over Kafka
        future = producer.send(topic, b"new_frame", timestamp_ms=round(time.time() * 1000))

        # Stop loop
        if event.is_set():
            break

    cv2.destroyAllWindows()
    ffmpeg_cmd.stdout.close()
    ffmpeg_cmd.wait()

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
