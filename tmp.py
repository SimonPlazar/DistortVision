import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("dnn\\yolov8s-seg.pt")

image = cv2.imread("input\\unknown.png")
image2 = cv2.flip(image, 1)

results = model.predict(source=[image, image2], conf=0.59, classes=0, verbose=False)

mask1 = results[0].masks.data[0].numpy()
mask2 = results[1].masks.data[0].numpy()

both = cv2.bitwise_or(mask1, mask2)

# mask_count = results.data.shape[1]
# mask = results.data[0].numpy()

cv2.imshow("Mask1", mask1)
cv2.imshow("Mask2", mask2)
cv2.imshow("Both", both)

cv2.waitKey(0)
cv2.destroyAllWindows()

