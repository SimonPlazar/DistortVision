import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("dnn\\yolov8n-seg.pt")

# Create the video capture object (replace '0' with the appropriate video source)
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model.predict(source=frame, conf=0.59, classes=0, verbose=False)[0].masks
    mask = (results.data[0].numpy() * 255).astype(np.uint8)
    print("Time taken: {:.2f} seconds".format(time.time() - start))

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
