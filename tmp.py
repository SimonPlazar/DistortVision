import cv2
import numpy as np
import time

if __name__ == "__main__":
    # Create the video capture object (replace '0' with the appropriate video source)
    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0)

    # ['FFMPEG', 'GSTREAMER', 'INTEL_MFX', 'MSMF', 'DSHOW', 'CV_IMAGES', 'CV_MJPEG', 'UEYE', 'OBSENSOR']

    # cap.set(cv2.CAP_PROP_FORMAT, -1)

    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Measure the processing time of the function
        start_time = time.time()

        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Processed Frame', frame)

        diff = time.time() - start_time
        print("Time taken: {:.2f} seconds".format(diff))
        print("FPS: {:.2f}".format(1 / diff))

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()
