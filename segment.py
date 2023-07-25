import time
import cv2
import numpy as np

# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("dnn\\frozen_inference_graph.pb",
                                    "dnn\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

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

    # Detect objects
    net.setInput(cv2.dnn.blobFromImage(frame, swapRB=True))

    boxes, masks = net.forward(["detection_out_final", "detection_masks"])

    for i in range(boxes.shape[2]):
        box = boxes[0, 0, i]
        class_id = box[1]
        if class_id != 0:
            continue
        score = box[2]
        if score < 0.5:
            continue

        # Get box Coordinates
        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)

        roi = black_image[y: y2, x: x2]
        roi_height, roi_width, _ = roi.shape

        # Get the mask
        mask = masks[i, int(class_id)]
        mask = cv2.resize(mask, (roi_width, roi_height))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 3)

        # Get mask coordinates
        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.fillPoly(roi, [cnt], (255, 0, 0))

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
