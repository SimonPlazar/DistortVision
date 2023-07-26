import math
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO("dnn\\yolov8s-seg.pt")

image = cv2.imread("input\\unknown.png")
image2 = cv2.flip(image, 1)

results = model.predict(source=[image, image2], conf=0.59, classes=0, verbose=False, device=0)[0].cpu()

mask1 = results.masks.data[0].numpy()

cv2.imshow("Mask1", mask1)

cv2.waitKey(0)
cv2.destroyAllWindows()

