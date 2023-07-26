import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("dnn\\yolov8s-seg.pt")

image = cv2.imread("input\\unknown.png")

results = model.predict(source=image, conf=0.59, classes=0, verbose=False)[0].masks

mask_count = results.data.shape[1]

for i in range(mask_count):
    mask = results.data[i].numpy()

    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(image.shape)
    print(mask.shape)

    np.repeat(results.data[0].numpy().reshape((height, width, 1)), 3, axis=-1).astype(np.uint8)

    extracted = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("extracted", extracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

