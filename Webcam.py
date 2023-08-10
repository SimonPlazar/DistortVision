import cv2
import cvzone
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

def censoring(extracted):
    image = Image.fromarray(cv2.cvtColor(cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV), cv2.COLOR_BGR2RGB))
    pixilated = image.resize((image.width // 7, image.height // 7), resample=Image.NEAREST)
    pixilated = pixilated.resize(image.size, resample=Image.NEAREST)
    return cv2.cvtColor(np.array(pixilated), cv2.COLOR_RGB2BGR)

def jpeg_artifact(extracted):
    _, compressed_image = cv2.imencode(".jpg", cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV), [int(cv2.IMWRITE_JPEG_QUALITY), 6])
    return cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

# Load the model
model = YOLO("dnn\\yolov8l-seg.pt")

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

height = 1080
width = 1920

capture.set(cv2.CAP_PROP_FPS, 60)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fpsReader = cvzone.FPS()

while True:
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)

    # Get the masks from the model
    results = model.predict(source=frame, conf=0.59, classes=0, verbose=False, device=0, retina_masks=True)[0].masks

    if results is not None:  # If a mask is found
        mask = (results.data[0] * 255).to(torch.uint8)

        for i in range(1, results.data.shape[0]):  # For each other mask
            mask = torch.bitwise_or(mask, (results.data[i] * 255).to(torch.uint8))  # Combine the masks

        # Pass the mask to cpu memory
        mask = mask.cpu().numpy()

        # Resize the mask to the video stream size
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Create dilated mask and extract the area defined by the mask from the original image
        dilated_mask = cv2.dilate(mask, None, iterations=5)
        extracted = cv2.bitwise_and(frame, frame, mask=dilated_mask)

        # Colorspace change
        # extracted = np.full((height, width, 3), (203,192,255), dtype=np.uint8) # Pink
        # extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV) # Convert to HSV

        # Jpeg artifact effect
        # extracted = jpeg_artifact(extracted)

        # Pixelated effect (censoring effect)
        extracted = censoring(extracted)

        # Use bitwise operations to replace the pixels in the original image
        roi = cv2.bitwise_and(extracted, extracted, mask=mask)
        inverse_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

        frame = cv2.add(background, roi)
    else: # If no mask is found
        cv2.putText(frame, "No mask found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Display a message

    fpsReader.update(frame, pos=(50, 80), color=(0, 255, 0), scale=5, thickness=5)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
