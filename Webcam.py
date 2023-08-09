import cv2
import cvzone

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

capture.set(cv2.CAP_PROP_FPS, 60)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fpsReader = cvzone.FPS()

while True:
    ret, frame = capture.read()
    fpsReader.update(frame, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()
