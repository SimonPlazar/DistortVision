import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from ultralytics import YOLO

def get_squere_mask(width, height):
    # Create a binary square mask (you can change this according to your needs)
    mask = np.zeros((height, width), np.uint8)
    x, y, w, h = width // 4, height // 4, width // 2, height // 2
    mask[y:y + h, x:x + w] = 255
    return mask

def compress_and_overlay(image, mask, quality):
    # Create a mask slightly larger than the input mask
    dilated_mask = cv2.dilate(mask, None, iterations=5)

    # Extract the area defined by the larger mask from the original image
    extracted = cv2.bitwise_and(image, image, mask=dilated_mask)

    # Compress the extracted area
    _, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    # Decode the compressed image
    extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

    # Replace the pixels in the original image with those from 'extracted',
    # but only for the pixels within the smaller mask.
    image[np.where(mask == 255)] = extracted[np.where(mask == 255)]

    return image


if __name__ == "__main__":
    # Load the model
    model = YOLO("dnn\\yolov8x-seg.pt")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(0)

    # MAGIC!!!
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    #
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('x', 'v', 'i', 'd'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('X', 'V', 'I', 'D'))
    cap.set(cv2.CAP_PROP_FPS, 30.0)


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get the width and height of the video stream
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Time and FPS variables
    start_time_fps = time.time()
    frame_counter = 0
    fps = 0

    while True:
        # Measure the processing time of the function
        start_time = time.time()

        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        # Get the masks from the model
        results = model.predict(source=frame, conf=0.59, classes=0, verbose=False, device=0, retina_masks=True)[0].masks
        # results = model.predict(source=cv2.resize(frame, (640, 480)), conf=0.59, classes=0, verbose=False, device=0, retina_masks=True)[0].masks

        if results is not None: # If a mask is found
            # mask = np.multiply(results.data[0].numpy(), 255).astype(np.uint8) # First mask
            mask = (results.data[0] * 255).to(torch.uint8)

            for i in range(1, results.data.shape[0]): # For each other mask
                #mask = np.bitwise_or(mask, np.multiply(results.data[i].numpy(), 255).astype(np.uint8)) # Combine the masks
                mask = torch.bitwise_or(mask, (results.data[i] * 255).to(torch.uint8)) # Combine the masks

            # Pass the mask to cpu memory
            mask = mask.cpu().numpy()

            # Resize the mask to the video stream size
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create dilated mask and extract the area defined by the mask from the original image
            dilated_mask = cv2.dilate(mask, None, iterations=5)
            extracted = cv2.bitwise_and(frame, frame, mask=dilated_mask)

            # Pink
            # extracted = np.full((height, width, 3), (203,192,255), dtype=np.uint8)

            # Alter extracted image
            # extracted = cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV)
            _, compressed_image = cv2.imencode(".jpg", extracted, [int(cv2.IMWRITE_JPEG_QUALITY), 6])
            extracted = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)

            # Replace the pixels in the original image with those from 'extracted',
            frame[np.where(mask == 255)] = extracted[np.where(mask == 255)]

        else: # If no mask is found
            cv2.putText(frame, "No mask found", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # Display a message

        # Calculate the FPS
        frame_counter += 1
        if (time.time() - start_time_fps) > 1:
            fps = frame_counter / (time.time() - start_time_fps)
            frame_counter = 0
            start_time_fps = time.time()

        # Display the processed frame with FPS information
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Processed Frame', frame)

        print("Time taken: {:.2f} seconds".format(time.time() - start_time))

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the windows
    cap.release()
    cv2.destroyAllWindows()
