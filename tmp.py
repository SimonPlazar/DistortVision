import cv2
import numpy as np
import time
from VideoCapture import Device

cam = Device()
cam.saveSnapshot('image.jpg')


if __name__ == "__main__":
    # Create the video capture object (replace '0' with the appropriate video source)
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        # Measure the processing time of the function
        start_time = time.time()

        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Processed Frame', frame)
        cv2.waitKey(5)

        diff = time.time() - start_time
        print("Time taken: {:.2f} seconds".format(diff))
        print("FPS: {:.2f}".format(1 / diff))

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()
