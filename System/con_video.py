import json
import numpy as np
import cv2
import threading
import redis
import kafka
import signal
from datetime import datetime

if __name__ == "__main__":
    # Redis
    red = redis.Redis()

    # Video
    frame = 0

    # Kafka
    topic = 'frame_noticifation'
    consumer = kafka.KafkaConsumer(bootstrap_servers='localhost:9092', auto_offset_reset='earliest',
                                   group_id='grp_visualization', consumer_timeout_ms=2000)
    consumer.subscribe([topic])

    while True:
        # Read from Redis when message is received over Kafka
        for message in consumer:
            if message.value.decode("utf-8") == "new_frame":
                frame_time = datetime.fromtimestamp(message.timestamp / 1000)
                curr_time = datetime.now()
                diff = (curr_time - frame_time).total_seconds()

                print("Time difference: {:.2f} seconds".format(diff))

