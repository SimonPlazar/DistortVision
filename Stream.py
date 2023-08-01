import time

import cv2
import numpy as np

# Use this line to capture video from the webcam
cap = cv2.VideoCapture(0)

# ['FFMPEG', 'GSTREAMER', 'INTEL_MFX', 'MSMF', 'DSHOW', 'CV_IMAGES', 'CV_MJPEG', 'UEYE', 'OBSENSOR']

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    print("Time taken: {:.2f} seconds".format(time.time() - start_time))

cap.release()
cv2.destroyAllWindows()