import numpy as np
import cv2
import threading
import redis
import signal
import time
import kafka

if __name__ == "__main__":
    # Redis
    red = redis.Redis()

    # Video
    vc = cv2.VideoCapture(0)

    # Kafka
    topic = 'frame_noticifation'
    producer = kafka.KafkaProducer(bootstrap_servers='localhost:9092')

    while True:
        ret, frame = vc.read()

        # Jump back to the beginning of input
        if not ret:
            vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Add frame to redis
        red.set("frame:latest", np.array(frame).tobytes())

        # Send notification about new frame over Kafka
        future = producer.send(topic, b"new_frame",
                               timestamp_ms=round(time.time() * 1000))

        # Wait until message is delivered to Kafka
        try:
            rm = future.get(timeout=10)
        except kafka.KafkaError:
            pass

