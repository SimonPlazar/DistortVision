import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("dnn\\yolov8s.pt")

# Create the video capture object (replace '0' with the appropriate video source)
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create black image
black_image = np.zeros((height, width, 3), np.uint8)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.59, classes=[0], verbose=False)[0].numpy()

    # Draw each box from results directly
    for param in results:
        for box in param.boxes.data:
            box_size = math.dist((box[0], box[1]), (box[2], box[3]))
            cv2.putText(frame, "Person " + str(round(box[4], 3)),
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if box_size > 180:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 0, 255), 2)
            elif box_size > 100:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)

    cv2.imshow("Black Image", black_image)
    cv2.imshow('Processed Frame', frame)

    # Create black image
    black_image = np.zeros((height, width, 3), np.uint8)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
